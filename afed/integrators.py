"""
.. module:: integrators
   :platform: Unix, Windows
   :synopsis: Adiabatic Free Energy Dynamics Integrators

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _Context: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Context.html
.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html
.. _CustomIntegrator: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomIntegrator.html
.. _Force: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.Force.html
.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html

"""

import re

from simtk import openmm, unit


class CustomIntegrator(openmm.CustomIntegrator):
    """
    An extension of OpenMM's CustomIntegrator_ class which facilitates the specification of
    both variables and computation steps in a per-driver-parameter fashion. These computations
    are defined in the same manner as per-dof computations in the original class.

    Parameters
    ----------
        stepSize : unit.Quantity
            The step size with which to integrate the system.
        drivingForce : :class:`~afed.afed.DrivingForce`
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

    def addPerParameterVariable(self, name, initialValue):
        """
        Defines a new per-driver-parameter variable.

        Parameters
        ----------
            variable : str
                The name of the per-driver-parameter variable.
            initialValue : unit.Quantity
                The value initially assigned to the new variable, for all driver parameters.

        """

        for parameter in self._driving_force._driver_parameters:
            self.addGlobalVariable(f'{name}_{parameter._name}', initialValue)
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


class MassiveMiddleSchemeIntegrator(CustomIntegrator):
    """
    An abstract class aimed at facilitating the implementation of different AFED integrators that
    differ on the employed thermostat but share the following features:

    1. Integration of particle-related degrees of freedom is done by using a middle-type scheme
    :cite:`Zhang_2017` (i.e. kick-move-bath-move-kick), possibly involving multiple time stepping
    (RESPA) :cite:`Tuckerman_1992`, and the system does not have any holonomic constraints.

    2. Integration of driver parameters and their attached thermostats is done with a middle-type
    scheme as well, and is detached from the integration of all other dynamic variables, including
    the driver-parameter velocities.

    3. Integration of driver-parameter velocities is always done along with the integration of
    particle velocities, with possible splitting into multiple time scales.

    Activation of multiple time scale integration (RESPA) is done by passing a list of integers
    through the keyword argument ``respaLoops`` (see below). The size of this list determines the
    number of considered time scales. Among the Force_ objects that belong to the simulated System_,
    only those whose force groups have been set to `k` will be considered at time scale `ḱ`. This
    includes the AFED-related :class:`~afed.afed.DrivingForce`.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath which the particles are attached to.
        stepSize : unit.Quantity
            The step size with which to integrate the system.
        drivingForce : :class:`~afed.afed.DrivingForce`
            The AFED driving force.

    Keyword Args
    ------------
        respaLoops : list(int), default=None
            A list of N integers, where ``respaLoops[k]`` determines how many iterations at time
            scale `k` are internally executed for every iteration at time scale `k+1`. If this is
            ``None``, then integration will take place at a single time scale.
        parameterLoops : int, default = 1
            The number of loops with which to subdivide the integration of driver parameters and
            their attached thermostats.

    """

    def __init__(self, temperature, stepSize, drivingForce, respaLoops=None, parameterLoops=1):
        super().__init__(stepSize, drivingForce)
        kT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temperature
        self.addGlobalVariable('kT', kT)
        self._respaLoops = respaLoops
        if respaLoops is not None:
            for scale, nsteps in enumerate(respaLoops):
                if nsteps > 1:
                    self.addGlobalVariable(f'irespa{scale}', 0)
        self._parameterLoops = parameterLoops
        if parameterLoops > 1:
            self.addGlobalVariable('iparam', 0)
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

    def _move(self, fraction, addCompute):
        addCompute('x', f'x + {fraction}*dt*v')

    def _kick(self, fraction, addCompute, group=''):
        addCompute('v', f'v + {fraction}*dt*f{group}/m')

    def _bath(self, fraction, addCompute):
        pass

    def addIntegrateParticles(self, fraction):
        if self._respaLoops is None:
            self._inner_loop(fraction, group='')
        else:
            self._integrate_particles_respa(fraction, len(self._respaLoops)-1)

    def addIntegrateParameters(self, fraction):
        n = self._parameterLoops
        if n > 1:
            self.addComputeGlobal('iparam', '0')
            self.beginWhileBlock(f'iparam < {n}')
        self._move(0.5*fraction/n, self.addComputePerParameter)
        self._bath(fraction/n, self.addComputePerParameter)
        self._move(0.5*fraction/n, self.addComputePerParameter)
        if n > 1:
            self.addComputeGlobal('iparam', 'iparam + 1')
            self.endBlock()


class MassiveMiddleNHCIntegrator(MassiveMiddleSchemeIntegrator):
    """
    An AFED integrator based on the massive Nosé-Hoover Chain thermostat :cite:`Martyna_1992`. This
    means that an independent thermostat chain is attached to each degree of freedom, including the
    AFED driver parameters. In this implementation, each chain is composed of two thermostats in
    series.

    All other properties of this integrator are inherited from :class:`MassiveMiddleSchemeIntegrator`.

    Parameters
    ----------
        temperature : unit.Quantity
            The temperature of the heat bath which the particles are attached to.
        timeScale : unit.Quantity
            The characteristic time scale of the thermostat chains.
        stepSize : unit.Quantity
            The step size with which to integrate the system.
        drivingForce : :class:`~afed.afed.DrivingForce`
            The AFED driving force.

    Keyword Args
    ------------
        respaLoops : list(int), default=None
            See :class:`MassiveMiddleSchemeIntegrator`.
        parameterLoops : int, default = 1
            See :class:`MassiveMiddleSchemeIntegrator`.

    """

    def __init__(self, temperature, timeScale, stepSize, drivingForce, **kwargs):
        super().__init__(temperature, stepSize, drivingForce, **kwargs)
        kT = unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*temperature
        for i in range(2):
            self.addGlobalVariable(f'Q{i+1}', kT*timeScale**2)
            self.addPerDofVariable(f'v{i+1}', 0)
            self.addPerParameterVariable(f'v{i+1}', 0)

        self.addIntegrateParticles(0.5)
        self.addIntegrateParameters(1)
        self.addIntegrateParticles(0.5)

    def _bath(self, fraction, addCompute):
        addCompute('v2', f'v2 + {fraction/2}*dt*(Q1*v1^2 - kT)/Q2')
        addCompute('v1', f'v1*exp(-{fraction/2}*dt*v2)')
        addCompute('v1', f'v1 + {fraction/2}*dt*(m*v^2 - kT)/Q1')
        addCompute('v', f'v*exp(-{fraction}*dt*v1)')
        addCompute('v1', f'v1 + {fraction/2}*dt*(m*v^2 - kT)/Q1')
        addCompute('v1', f'v1*exp(-{fraction/2}*dt*v2)')
        addCompute('v2', f'v2 + {fraction/2}*dt*(Q1*v1^2 - kT)/Q2')
