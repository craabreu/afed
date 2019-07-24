"""
afed.py
Adiabatic Free Energy Dynamics with OpenMM

Handles the primary functions
"""
import copy

from simtk import openmm, unit


class DrivenCollectiveVariable(object):
    """
    A collective variable whose dynamics is meant to be driven by an extended-space variable.

    .. _Force: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Force.html

    Parameters
    ----------
        name : str
            The name of the collective variable.
        variable : openmm.Force
            An OpenMM Force_ object whose energy function is used to evaluate the collective
            variable for a given Context.
        dimension : unit.Unit
            The unit of measurement with which the collective variable is returned by the Force_
            object specified as `variable` (see above). This is supposed to be one of the base
            units employed by OpenMM (see here_).
            .. _here: http://docs.openmm.org/latest/userguide/theory.html#units
        period : unit.Quantity, optional, default=None
            The period of a periodic variable, which must bear a unit of measurement compatible
            with `unit` (see above). If this is `None`, then the variable is considered to be
            aperiodic.

    Example
    -------
        >>> from simtk import openmm, unit
        >>> cv = openmm.CustomTorsionForce('theta')
        >>> cv.addTorsion(0, 1, 2, 3, [])
        >>> psi = DrivenCollectiveVariable('psi', cv, unit=unit.radians, period=360*unit.degrees)
        >>> print(psi)
        psi, dimension=radian, period=6.283185307179586 rad

    """
    def __init__(self, name, variable, dimension, period=None):
        self._name = name
        self._variable = variable
        self._dimension = dimension
        self._period = None if period is None else period.value_in_unit(dimension)

    def __repr__(self):
        return f'{self._name}, dimension={self._dimension}, period={self._period}'

    def evaluate(self, positions, box_vectors=None):
        """
        Computes the value of the collective variable for a given set of particle coordinates and
        box vectors. The use of periodic boundary conditions will depend on the Force object that
        defines the collective variable.

        Parameters
        ----------
            positions : list(openmm.Vec3)
                A list whose length equals the number of particles in the system.
            box_vectors : list(openmm.Vec3), optional, default=None
                A list with three vectors containing the edges of the simulation box.

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
    terminology of OpenMM, this variable is a context parameter.

    .. warning::
        The extended variable driving a periodic collective variable will be periodic as well,
        with the same period of that variable.

    Parameters
    ----------
        name : str
            The name of the driver extended-space variable.
        initial_value : unit.Quantity
            The initial value which the driver parameter must assume. The value must bear a unit of
            measurement compatible with the associated driven variable.
        temperature : unit.Quantity
            The temperature (:math:`T`) of the heat bath attached to the driver parameter.
        velocity_scale : unit.Quantity
            The characteristic velocity scale (:math:`\\nu`) of the driver parameter in units
            compatible with `driven-variable-units/time`. The inertial mass of the driver parameter
            is computed as :math:`k_B T/\\nu^2`, where :math:`k_B` is the Boltzmann contant.
        lower_bound : unit.Quantity
            The lower limit imposed to the driver variable by means of a hard wall. If this is set
            to `None`, then the variable will not be intentionally bounded from below.
        upper_bound : unit.Quantity
            The upper limit imposed to the driver variable by means of a hard wall. If this is set
            to `None`, then the variable will not be intentionally bounded from above.
        periodic : bool, optional, default=False
            Whether this driver parameter is periodic with `period = upper_bound - lower_bound`.

    """
    def __init__(self, name, dimension, initial_value, temperature, velocity_scale,
                 lower_bound, upper_bound, periodic=False):
        self._name = name
        self._initial_value = initial_value
        self._dimension = initial_value.unit if isinstance(initial_value, unit.Quantity) else None
        kB = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        self._kT = kB*temperature
        self._mass = self._kT/velocity_scale**2
        if self._dimension is None:
            self._lower_bound = lower_bound
            self._upper_bound = upper_bound
        else:
            self._lower_bound = None if lower_bound is None else lower_bound/self._dimension
            self._upper_bound = None if upper_bound is None else upper_bound/self._dimension
        self._period = self._upper_bound - self._lower_bound if periodic else None

    def __repr__(self):
        return f'{self._name}, initial value={self._initial_value}, kT={self._kT}, mass={self._mass}'


class HarmonicDrivingForce(openmm.CustomCVForce):
    """
    The typical harmonic driving potential used in the driven Adiabatic Free Energy Dynamics (dAFED)
    method.

    """
    def __init__(self):
        super().__init__('')
        self._energy_terms = []
        self._driver_parameters = []

    def __repr__(self):
        return self.getEnergyFunction()

    def add(self, variable, parameter, force_constant):
        """
        Adds a pair of driven variable and driver parameter.

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
            self._energy_terms.append(f'{0.5*K}*min({delta},{variable._period}-{delta})^2')
        else:
            self._energy_terms.append(f'{0.5*K}*({variable._name}-{parameter._name})^2')
        self.setEnergyFunction('+'.join(self._energy_terms))
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
                expression = 'y - L*floor((y - ymin)/L)'
                expression += f'; y = {name} + {fraction}*dt*v_{name}'
                expression += f'; ymin = {parameter._lower_bound}'
                expression += f'; L = {parameter._upper_bound - parameter._lower_bound}'
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
