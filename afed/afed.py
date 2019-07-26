"""
.. module:: afed
   :platform: Unix, Windows
   :synopsis: Adiabatic Free Energy Dynamics with OpenMM

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _Context: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Context.html
.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html
.. _CustomIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html
.. _Force: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Force.html
.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html

"""

import copy
import re

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

    def evaluate(self, positions, boxVectors=None):
        """
        Computes the value of the collective variable for a given set of particle coordinates
        and box vectors. Whether periodic boundary conditions will be used or not depends on
        the corresponding attribute of the Force_ object specified as the collective variable.

        Parameters
        ----------
            positions : list(openmm.Vec3)
                A list whose length equals the number of particles in the system and which contains
                the coordinates of these particles.
            boxVectors : list(openmm.Vec3), optional, default=None
                A list with three vectors which describe the edges of the simulation box.

        Returns
        -------
            unit.Quantity

        """

        system = openmm.System()
        for i in range(len(positions)):
            system.addParticle(0)
        if boxVectors is not None:
            system.setDefaultPeriodicBoxVectors(*boxVectors)
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
        initialValue : unit.Quantity
            The initial value which the driver parameter must assume. It must bear a unit of
            measurement compatible with the specified ``dimension``.
        temperature : unit.Quantity
            The temperature of the heat bath which the driver parameter will be attached to.
        velocityScale : unit.Quantity
            The characteristic velocity scale (:math:`\\nu`) of the driver parameter. It must bear
            a unit of measurement compatible with ``dimension``/time. The inertial mass of the
            driver parameter will be computed as :math:`k_B T/\\nu^2`, where :math:`k_B` is the
            Boltzmann contant and :math:`T` is ``temperature``.
        lowerBound : unit.Quantity, optional, default=None
            The lower limit imposed to the driver parameter by means of a hard wall or periodic
            boundary conditions. If this is ``None``, then the parameter will not be intentionally
            bounded from below. Otherwise, the argument must bear a unit of measurement compatible
            with ``dimension``.
        upperBound : unit.Quantity, optional, default=None
            The upper limit imposed to the driver parameter by means of a hard wall or periodic
            boundary conditions. If this is ``None``, then the parameter will not be intentionally
            bounded from above. Otherwise, the argument must bear a unit of measurement compatible
            with ``dimension``.
        periodic : bool, optional, default=False
            Whether the driver parameter is a periodic quantity with period equal to the difference
            between ``upperBound`` and ``lowerBound``.

    """

    def __init__(self, name, dimension, initialValue, temperature, velocityScale,
                 lowerBound=None, upperBound=None, periodic=False):
        self._name = name
        self._initial_value = initialValue
        self._dimension = dimension
        kB = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA
        self._kT = kB*temperature
        self._mass = self._kT/velocityScale**2
        self._lower_bound = lowerBound
        self._upper_bound = upperBound
        self._periodic = periodic
        if periodic and (self._lower_bound is None or self._upper_bound is None):
            raise Exception('Bounds must be defined for a periodic driver parameter')

    def __repr__(self):
        return f'{self._name}, initial value={self._initial_value}, kT={self._kT}, mass={self._mass}'


class DrivingForce(openmm.CustomCVForce):
    """
    A extension of OpenMM's CustomCVForce_ class with special treatment for pairs of AFED-related
    pairs of :class:`DrivenCollectiveVariable` and :class:`DriverParameter`.

    Parameters
    ----------
        energy : str
            An algebraic expression giving a contribution to the system energy as a function of the
            driven collective variables and driver parameters, as well as other standard collective
            variables and global parameters.

    """

    def __init__(self, energy):
        super().__init__(energy)
        self._driver_parameters = []

    def __repr__(self):
        return self.getEnergyFunction()

    def addPair(self, variable, parameter):
        """
        Adds a pair of driven collective variable and driver parameter.

        Parameters
        ----------
            variable : :class:`DrivenCollectiveVariable`
                The driven collective variable.
            parameter : :class:`DriverParameter`
                The driver parameter.

        """

        self._driver_parameters.append(parameter)
        self.addCollectiveVariable(variable._name, variable._variable)
        self.addGlobalParameter(parameter._name, parameter._initial_value)
        self.addEnergyParameterDerivative(parameter._name)


class HarmonicDrivingForce(DrivingForce):
    """
    A special case of :class:`DrivingForce` to handle the typical harmonic driving potential used
    in the driven Adiabatic Free Energy Dynamics (dAFED) method.

    """

    def __init__(self):
        super().__init__('')
        self._energy_terms = []

    def addPair(self, variable, parameter, forceConstant):
        """
        Adds a pair of driven collective variable and driver parameter and specifies the force
        constant of their harmonic coupling.

        Parameters
        ----------
            variable : :class:`DrivenCollectiveVariable`
                The driven collective variable.
            parameter : :class:`DriverParameter`
                The driver parameter.
            forceConstant : unit.Quantity
                The strength of the coupling harmonic force in units of energy per squared dimension
                of the collective variable.

        """

        K = forceConstant.value_in_unit(unit.kilojoules_per_mole/variable._dimension**2)
        if variable._period is not None:
            delta = f'abs({variable._name}-{parameter._name})'
            period = variable._period/variable._dimension
            self._energy_terms.append(f'{0.5*K}*min({delta},{period}-{delta})^2')
        else:
            self._energy_terms.append(f'{0.5*K}*({variable._name}-{parameter._name})^2')
        self.setEnergyFunction('+'.join(self._energy_terms))
        super().addPair(variable, parameter)


class CustomIntegrator(openmm.CustomIntegrator):
    """
    An extension of OpenMM's CustomIntegrator_ class which facilitates the specification of
    both variables and computation steps in a per-driver-parameter fashion. These computations
    are defined in the same manner as per-dof computations in the original class.

    Parameters
    ----------
        stepSize : unit.Quantity
            The step size with which to integrate the system.
        drivingForce : :class:`DrivingForce`
            The AFED driving force.

    """

    def __init__(self, stepSize, drivingForce):
        super().__init__(stepSize)
        self._driving_force = drivingForce
        self._per_parameter_variables = set(['v', 'm', 'kT'])
        for parameter in drivingForce._driver_parameters:
            self.addGlobalVariable(f'v_{parameter._name}', 0)
            self.addGlobalVariable(f'm_{parameter._name}', parameter._mass)
            self.addGlobalVariable(f'kT_{parameter._name}', parameter._kT)

    def __repr__(self):
        # Human-readable version of each integrator step (adapted from choderalab/openmmtools)
        step_type_str = [
            '{target} <- {expr}',
            '{target} <- {expr}',
            '{target} <- sum({expr})',
            'constrain positions',
            'constrain velocities',
            'allow forces to update the context state',
            'if ({expr}):',
            'while ({expr}):',
            'end'
        ]
        readable_lines = []
        indent_level = 0
        for step in range(self.getNumComputations()):
            line = ''
            step_type, target, expr = self.getComputationStep(step)
            if step_type == 8:
                indent_level -= 1
            command = step_type_str[step_type].format(target=target, expr=expr)
            line += '{:4d}: '.format(step) + ' '*3*indent_level + command
            if step_type in [6, 7]:
                indent_level += 1
            readable_lines.append(line)
        return '\n'.join(readable_lines)

    def addPerParameterVariable(self, name, value):
        for parameter in self._driving_force._driver_parameters:
            self.addGlobalVariable(f'{name}_{parameter._name}', value)
        self._per_parameter_variables.add(name)

    def addComputePerParameter(self, variable, expression):
        """
        Add a step to the integration algorithm that computes a per-driver-parameter value.

        Parameters
        ----------
            variable : str
                The per-driver-parameter variable to store the computed value into.
            expression : str
                A mathematical expression involving both global and per-driver-parameter variables.
                In each integration step, its value is computed for every driver parameter and
                stored into the specified variable.

        Returns
        -------
            The index of the last step that was added.

        """

        def translate(expression, parameter):
            output = re.sub(r'\bx\b', f'{parameter}', expression)
            output = re.sub(r'\bf([0-9]*)\b', f'(-deriv(energy\\1,{parameter}))', output)
            for symbol in self._per_parameter_variables:
                output = re.sub(r'\b{}\b'.format(symbol), f'{symbol}_{parameter}', output)
            return output

        for parameter in self._driving_force._driver_parameters:
            name = parameter._name
            if variable == 'x':
                if parameter._periodic:
                    # Apply periodic boundary conditions:
                    ymin = parameter._lower_bound/parameter._dimension
                    ymax = parameter._upper_bound/parameter._dimension
                    corrected_expression = f'y - L*floor((y - ymin)/L)'
                    corrected_expression += f'; ymin = {ymin}'
                    corrected_expression += f'; L = {ymax - ymin}'
                    corrected_expression += f'; y = {translate(expression, name)}'
                    index = self.addComputeGlobal(name, corrected_expression)
                else:
                    # Apply ellastic collision with hard wall:
                    index = self.addComputeGlobal(name, translate(expression, name))
                    for bound, op in zip([parameter._lower_bound, parameter._upper_bound], ['<', '>']):
                        if bound is not None:
                            limit = bound/parameter._dimension
                            self.beginIfBlock(f'{name} {op} {limit}')
                            self.addComputeGlobal(name, f'{2*limit}-{name}')
                            index = self.addComputeGlobal(f'v_{name}', f'-v_{name}')
                            self.endBlock()
            elif variable in self._per_parameter_variables:
                index = self.addComputeGlobal(f'{variable}_{name}', translate(expression, name))
            else:
                raise Exception('invalid per-parameter variable')
        return index


class MiddleSchemeAFEDIntegrator(CustomIntegrator):
    """
    An abstract class aimed at facilitating the implementation of different AFED integrators that
    differ on the employed thermostat but share the following features:

    1. Integration of particle-related degrees of freedom is done by using a middle-type scheme
    (i.e. kick-move-bath-move-kick), possibly involving multiple time stepping (RESPA), and the
    system does not have any holonomic constraints.

    2. Integration of driver parameters and their attached thermostats is done with a middle-type
    scheme as well, and is detached from the integration of all other dynamic variables, including
    the driver-parameter velocities.

    3. Integration of driver-parameter velocities is always done along with the integration of
    particle velocities, with possible splitting into multiple time scales.

    Activation of multiple time scale integration (RESPA) is done by passing a list of integers
    through the keyword argument ``respaLoops`` (see below). The size of this list determines the
    number of considered time scales. Among the Force_ objects that belong to the simulated System_,
    only those whose force groups have been set to `k` will be considered at time scale `ḱ`. This
    includes the AFED-related :class:`DrivingForce`.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath which the particles are attached to.
        stepSize : unit.Quantity
            The step size with which to integrate the system.
        drivingForce : :class:`DrivingForce`
            The AFED driving force.
        respaLoops : list(int), optional, default=None
            A list of N integers, where ``respaLoops[k]`` determines how many iterations at time
            scale `k` are internally executed for every iteration at time scale `k+1`. If this is
            ``None``, then integration will take place at a single time scale.

    """

    def __init__(self, temperature, stepSize, drivingForce, respaLoops):
        super().__init__(stepSize, drivingForce)
        kT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temperature
        self.addGlobalVariable('kT', kT)
        self._respaLoops = respaLoops
        if respaLoops is not None:
            for scale, nsteps in enumerate(respaLoops):
                if nsteps > 1:
                    self.addGlobalVariable(f'irespa{scale}', 0)
        self.addThermostat(lambda fraction, compute: None)
        self.addUpdateContextState()

    def _integrate_particles_respa(self, fraction, scale):
        n = self._respaLoops[scale]
        if n > 1:
            self.addComputeGlobal(f'irespa{scale}', '0')
            self.beginWhileBlock(f'irespa{scale} < {n}')
        if scale > 0:
            self._kick(fraction/(2*n), self.addComputePerParameter, scale)
            self._kick(fraction/(2*n), self.addComputePerDof, scale)
            self._integrate_particles_respa(fraction/n, scale-1)
            self._kick(fraction/(2*n), self.addComputePerDof, scale)
            self._kick(fraction/(2*n), self.addComputePerParameter, scale)
        else:
            self._inner_loop(fraction/n, group=0)
        if n > 1:
            self.addComputeGlobal(f'irespa{scale}', f'irespa{scale} + 1')
            self.endBlock()

    def _inner_loop(self, fraction, group):
        self._kick(fraction/2, self.addComputePerParameter, group)
        self._kick(fraction/2, self.addComputePerDof, group)
        self._move(fraction/2, self.addComputePerDof)
        self._bath(fraction, self.addComputePerDof)
        self._move(fraction/2, self.addComputePerDof)
        self._kick(fraction/2, self.addComputePerDof, group)
        self._kick(fraction/2, self.addComputePerParameter, group)

    def _move(self, fraction, compute):
        compute('x', f'x + {fraction}*dt*v')

    def _kick(self, fraction, compute, group=''):
        compute('v', f'v + {fraction}*dt*f{group}/m')

    def addThermostat(self, function):
        self._bath = function

    def addIntegrateParticles(self, fraction):
        if self._respaLoops is None:
            self._inner_loop(fraction, group='')
        else:
            self._integrate_particles_respa(fraction, len(self._respaLoops)-1)

    def addIntegrateParameters(self, fraction):
        self._move(0.5, self.addComputePerParameter)
        self._bath(1.0, self.addComputePerParameter)
        self._move(0.5, self.addComputePerParameter)


class MassiveNHCIntegrator(MiddleSchemeAFEDIntegrator):
    """
    An AFED integrator based on the massive Nosé-Hoover Chain thermostat. This means that an
    independent thermostat chain is attached to each degree of freedom, including the AFED driver
    parameters. In this implementation, each chain is composed of two thermostats in series.

    All other properties of this integrator are inherited from :class:`MiddleSchemeAFEDIntegrator`.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath which the particles are attached to.
        timeScale : unit.Quantity
            The characteristic time scale of the thermostat chains.
        stepSize : unit.Quantity
            The step size with which to integrate the system.
        drivingForce : :class:`DrivingForce`
            The AFED driving force.
        respaLoops : list(int), optional, default=None
            See :class:`MiddleSchemeAFEDIntegrator`.

    """
    def __init__(self, temperature, timeScale, stepSize, drivingForce, respaLoops=None):
        super().__init__(temperature, stepSize, drivingForce, respaLoops)
        kT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temperature
        for i in range(2):
            self.addGlobalVariable(f'Q{i+1}', kT*timeScale**2)
            self.addPerDofVariable(f'v{i+1}', 0)
            self.addPerParameterVariable(f'v{i+1}', 0)

        def NoseHooverChain(fraction, compute):
            compute('v2', f'v2 + {fraction/2}*dt*(Q1*v1^2 - kT)/Q2')
            compute('v1', f'v1*exp(-{fraction/2}*dt*v2)')
            compute('v1', f'v1 + {fraction/2}*dt*(m*v^2 - kT)/Q1')
            compute('v', f'v*exp(-{fraction}*dt*v1)')
            compute('v1', f'v1 + {fraction/2}*dt*(m*v^2 - kT)/Q1')
            compute('v1', f'v1*exp(-{fraction/2}*dt*v2)')
            compute('v2', f'v2 + {fraction/2}*dt*(Q1*v1^2 - kT)/Q2')

        self.addThermostat(NoseHooverChain)
        self.addIntegrateParticles(0.5)
        self.addIntegrateParameters(1)
        self.addIntegrateParticles(0.5)


class BAOABIntegrator(CustomIntegrator):
    def __init__(self, temperature, frictionCoeff, stepSize, drivingForce, numRattles=0):
        super().__init__(stepSize, drivingForce)
        self._rattles = numRattles
        kT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temperature
        self.addGlobalVariable('kT', kT)
        self.addGlobalVariable('friction', frictionCoeff)
        self._rattles > 1 and self.addGlobalVariable('irattle', 0)
        self._rattles > 0 and self.addPerDofVariable('x0', 0)
        self.addUpdateContextState()
        self._B(0.5)
        self._A(0.5)
        self._O(1)
        self._A(0.5)
        self._B(0.5)

    def _A(self, fraction):
        if self._rattles > 1:
            self.addComputeGlobal('irattle', '0')
            self.beginWhileBlock(f'irattle < {self._rattles}')
        self.addComputePerDof('x', f'x + {fraction/max(1, self._rattles)}*dt*v')
        if self._rattles > 0:
            self.addComputePerDof('x0', 'x')
            self.addConstrainPositions()
            self.addComputePerDof('v', f'v + (x - x0)/({fraction/max(1, self._rattles)}*dt)')
            self.addConstrainVelocities()
        if self._rattles > 1:
            self.addComputeGlobal('irattle', 'irattle + 1')
            self.endBlock()
        self.addComputePerParameter('x', f'x + {fraction}*dt*v')

    def _B(self, fraction):
        self.addComputePerDof('v', f'v + {fraction}*dt*f/m')
        self._rattles > 0 and self.addConstrainVelocities()
        self.addComputePerParameter('v', f'v + {fraction}*dt*f/m')

    def _O(self, fraction):
        expression = 'z*v + sqrt((1 - z*z)*kT/m)*gaussian'
        expression += f'; z = exp(-{fraction}*friction*dt)'
        self.addComputePerDof('v', expression)
        self._rattles > 0 and self.addConstrainVelocities()
        self.addComputePerParameter('v', expression)
