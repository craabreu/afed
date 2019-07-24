"""
.. module:: afed
   :platform: Unix, Windows
   :synopsis: Adiabatic Free Energy Dynamics with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _Force: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Force.html
.. _Context: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Context.html

"""

import copy

from simtk import openmm, unit


class DrivenCollectiveVariable(object):
    """
    A collective variable whose dynamics is meant to be driven by an extended-space variable.

    Parameters
    ----------
        name : str
            The name of the collective variable.
        variable : openmm.Force
            An OpenMM Force_ object whose energy function is used to evaluate the collective
            variable for a given Context_.
        dimension : unit.Unit
            The unit of measurement with which the collective variable is returned by the Force_
            object specified as ``variable``. If the variable is a dimensionless quantity, then one
            must explicitly state it by entering ``unit.dimensionless``. Otherwise, the dimension
            is supposed to be one of the base units employed by OpenMM (see `here
            <http://docs.openmm.org/latest/userguide/theory.html#units>`_) or a combination thereof.
        period : unit.Quantity, optional, default=None
            The period of the collective variable if it is periodic. This argument must bear a unit
            of measurement compatible with the specified ``dimension``. If the argument is ``None``,
            then the collective variable is considered to be aperiodic.

    Example
    -------
        >>> from simtk import openmm, unit
        >>> cv = openmm.CustomTorsionForce('theta')
        >>> cv_index = cv.addTorsion(0, 1, 2, 3, [])
        >>> psi = DrivenCollectiveVariable('psi', cv, unit.radians, period=360*unit.degrees)
        >>> print(psi)
        psi, dimension=radian, period=360 degrees

    """

    def __init__(self, name, variable, dimension=unit.dimensionless, period=None):
        self._name = name
        self._variable = variable
        self._dimension = dimension
        self._period = period

    def __repr__(self):
        return f'{self._name}, dimension={self._dimension}, period={self._period}'

    def evaluate(self, positions, box_vectors=None):
        """
        Computes the value of the collective variable for a given set of particle coordinates
        and box vectors. Whether periodic boundary conditions will be used or not depends on
        the corresponding attribute of the Force_ object specified as the collective variable.

        Parameters
        ----------
            positions : list(openmm.Vec3)
                A list whose length equals the number of particles in the system and which contains
                the coordinates of these particles.
            box_vectors : list(openmm.Vec3), optional, default=None
                A list with three vectors which describe the edges of the simulation box.

        Returns
        -------
            unit.Quantity

        """

        system = openmm.System()
        for i in range(len(positions)):
            system.addParticle(0)
        if box_vectors is not None:
            system.setDefaultPeriodicBoxVectors(*box_vectors)
        system.addForce(copy.deepcopy(self._variable))
        context = openmm.Context(system, openmm.CustomIntegrator(0))
        context.setPositions(positions)
        energy = context.getState(getEnergy=True).getPotentialEnergy()
        return energy.value_in_unit(unit.kilojoules_per_mole)*self._dimension


class DriverParameter(object):
    """
    An extended-space variable aimed at driving the dynamics of a collective variable. In the
    terminology of OpenMM, this variable is a Context_ parameter.

    Parameters
    ----------
        name : str
            The name of the driver parameter.
        dimension : unit.Unit
            The unit of measurement of this driver parameter. If it is a dimensionless quantity,
            then one must explicitly state it by entering ``unit.dimensionless``.
        initial_value : unit.Quantity
            The initial value which the driver parameter must assume. It must bear a unit of
            measurement compatible with the specified ``dimension``.
        temperature : unit.Quantity
            The temperature of the heat bath which the driver parameter will be attached to.
        velocity_scale : unit.Quantity
            The characteristic velocity scale (:math:`\\nu`) of the driver parameter. It must bear
            a unit of measurement compatible with ``dimension``/time. The inertial mass of the
            driver parameter will be computed as :math:`k_B T/\\nu^2`, where :math:`k_B` is the
            Boltzmann contant and :math:`T` is ``temperature``.
        lower_bound : unit.Quantity, optional, default=None
            The lower limit imposed to the driver parameter by means of a hard wall. If this is
            ``None``, then the parameter will not be intentionally bounded from below.
        upper_bound : unit.Quantity, optional, default=None
            The upper limit imposed to the driver parameter by means of a hard wall. If this is
            ``None``, then the parameter will not be intentionally bounded from above.

    """

    def __init__(self, name, dimension, initial_value, temperature, velocity_scale,
                 lower_bound=None, upper_bound=None):
        self._name = name
        self._initial_value = initial_value
        self._dimension = dimension
        kB = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        self._kT = kB*temperature
        self._mass = self._kT/velocity_scale**2
        self._lower_bound = lower_bound
        self._upper_bound = upper_bound

    def __repr__(self):
        return f'{self._name}, initial value={self._initial_value}, kT={self._kT}, mass={self._mass}'


class HarmonicDrivingForce(openmm.CustomCVForce):
    """
    The typical harmonic driving potential used in the driven Adiabatic Free Energy Dynamics (dAFED)
    method.

    .. warning::
        For every periodic collective variable, the corresponding driver parameter will also be
        considered to be periodic and have the same period.

    """

    def __init__(self):
        super().__init__('')
        self._energy_terms = []
        self._driver_parameters = []

    def __repr__(self):
        return self.getEnergyFunction()

    def add_pair(self, variable, parameter, force_constant):
        """
        Adds a pair of driven collective variable and driver parameter and specifies the force
        constant of their harmonic coupling.

        Parameters
        ----------
            variable : :class:`DrivenCollectiveVariable`
                The driven collective variable.
            parameter : :class:`DriverParameter`
                The driver parameter.
            force_constant : unit.Quantity
                The strength of the coupling harmonic force in units of energy per squared dimension
                of the collective variable.

        """
        self._driver_parameters.append(parameter)
        K = force_constant.value_in_unit(unit.kilojoules_per_mole/variable._dimension**2)
        if variable._period is not None:
            delta = f'abs({variable._name}-{parameter._name})'
            period = variable._period/variable._dimension
            self._energy_terms.append(f'{0.5*K}*min({delta},{period}-{delta})^2')
        else:
            self._energy_terms.append(f'{0.5*K}*({variable._name}-{parameter._name})^2')
        self.setEnergyFunction('+'.join(self._energy_terms))
        # TODO: check if collective variable and/or parameter have already been added:
        self.addCollectiveVariable(variable._name, variable._variable)
        self.addGlobalParameter(parameter._name, parameter._initial_value)
        self.addEnergyParameterDerivative(parameter._name)


class GeodesicBAOABIntegrator(openmm.CustomIntegrator):
    def __init__(self, timestep, temperature, friction_coefficient, number_of_rattles=1,
                 driving_force=None):
        super().__init__(timestep)
        self._driving_force = driving_force
        kT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temperature
        self.addGlobalVariable('kT', kT)
        self.addGlobalVariable('friction', friction_coefficient)
        self.addGlobalVariable('irattle', 0)
        self.addPerDofVariable('x0', 0)
        if driving_force is not None:
            for parameter in driving_force._driver_parameters:
                self.addGlobalVariable(f'v_{parameter._name}', 0)
                self.addGlobalVariable(f'm_{parameter._name}', parameter._mass)
                self.addGlobalVariable(f'kT_m_{parameter._name}', parameter._kT/parameter._mass)
        self.addUpdateContextState()
        self._B(0.5)
        self._A(0.5, number_of_rattles)
        self._O(1)
        self._A(0.5, number_of_rattles)
        self._B(0.5)

    def _A(self, fraction, rattles):
        if rattles > 1:
            self.addComputeGlobal('irattle', '0')
            self.beginWhileBlock(f'irattle < {rattles}')
        self.addComputePerDof('x', f'x + {fraction/rattles}*dt*v')
        self.addComputePerDof('x0', 'x')
        self.addConstrainPositions()
        self.addComputePerDof('v', f'v + (x - x0)/({fraction/rattles}*dt)')
        self.addConstrainVelocities()
        if rattles > 1:
            self.addComputeGlobal('irattle', 'irattle + 1')
            self.endBlock()
        if self._driving_force is not None:
            for parameter in self._driving_force._driver_parameters:
                name = parameter._name
                ymin = parameter._lower_bound/parameter._dimension
                ymax = parameter._upper_bound/parameter._dimension
                expression = f'y - L*floor((y - {ymin})/L)'
                expression += f'; y = {name} + {fraction}*dt*v_{name}'
                expression += f'; L = {ymax - ymin}'
                self.addComputeGlobal(name, expression)

    def _B(self, fraction):
        self.addComputePerDof('v', 'v + 0.5*dt*f/m')
        self.addConstrainVelocities()
        if self._driving_force is not None:
            for parameter in self._driving_force._driver_parameters:
                name = parameter._name
                expression = f'v_{name} - {fraction}*dt*deriv(energy,{name})/m_{name}'
                self.addComputeGlobal(f'v_{name}', expression)

    def _O(self, fraction):
        expression = 'z*v + sqrt((1 - z*z)*kT/m)*gaussian'
        z_definition = f'; z = exp(-{fraction}*friction*dt)'
        self.addComputePerDof('v', expression + z_definition)
        self.addConstrainVelocities()
        if self._driving_force is not None:
            for parameter in self._driving_force._driver_parameters:
                name = parameter._name
                expression = f'z*v_{name} + sqrt((1 - z*z)*kT_m_{name})*gaussian'
                self.addComputeGlobal(f'v_{name}', expression + z_definition)
