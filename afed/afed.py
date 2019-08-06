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
        >>> import afed
        >>> from simtk import openmm, unit
        >>> cv = openmm.CustomTorsionForce('theta')
        >>> cv_index = cv.addTorsion(0, 1, 2, 3, [])
        >>> psi = afed.DrivenCollectiveVariable('psi', cv, unit.radians, period=360*unit.degrees)
        >>> print(psi)
        psi, dimension=radian, period=360 deg

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

        Example
        -------
            >>> import afed
            >>> from simtk import unit
            >>> model = afed.AlanineDipeptideModel()
            >>> psi_angle, _ = model.getDihedralAngles()
            >>> psi = afed.DrivenCollectiveVariable('psi', psi_angle, unit.radians, period=360*unit.degrees)
            >>> psi.evaluate(model.getPositions())
            Quantity(value=3.141592653589793, unit=radian)

        """

        system = openmm.System()
        for i in range(len(positions)):
            system.addParticle(0)
        if boxVectors is not None:
            system.setDefaultPeriodicBoxVectors(*boxVectors)
        system.addForce(copy.deepcopy(self._variable))
        platform = openmm.Platform.getPlatformByName('Reference')
        context = openmm.Context(system, openmm.CustomIntegrator(0), platform)
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

    Example
    -------
        >>> import afed
        >>> from simtk import unit
        >>> model = afed.AlanineDipeptideModel()
        >>> psi_angle, _ = model.getDihedralAngles()
        >>> psi = afed.DrivenCollectiveVariable('psi', psi_angle, unit.radians, period=360*unit.degrees)
        >>> psi_value = psi.evaluate(model.getPositions())
        >>> afed.DriverParameter('psi_s', unit.radians, psi_value,
        ...                      1500*unit.kelvin, 0.003*unit.radians/unit.femtosecond,
        ...                      -180*unit.degrees, 180*unit.degrees, periodic=True)
        psi_s, initial value=3.141592653589793 rad

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
        return f'{self._name}, initial value={self._initial_value}'

    def getMass(self):
        """
        Gets the mass associated to the driver parameter.

        Returns
        -------
            mass : unit.Quantity

        Example
        -------
            >>> import afed
            >>> from simtk import unit
            >>> model = afed.AlanineDipeptideModel()
            >>> psi_driver, _ = model.getDriverParameters()
            >>> psi_driver.getMass()
            Quantity(value=1.6800000000000003e-21, unit=second**2*joule/(mole*radian**2))

        """

        return self._mass


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
        self._driven_variables = []
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

        self._driven_variables.append(variable)
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

        Example
        -------
            >>> import afed
            >>> from copy import deepcopy
            >>> from simtk import unit
            >>> model = afed.AlanineDipeptideModel()
            >>> psi, phi = model.getCollectiveVariables(copy=True)
            >>> psi_driver, phi_driver = model.getDriverParameters()
            >>> K = 2.78E3*unit.kilocalories_per_mole/unit.radians**2
            >>> force = afed.HarmonicDrivingForce()
            >>> force.addPair(psi, psi_driver, K)
            >>> print(force)
            5815.76*min(abs(psi-psi_s),6.283185307179586-abs(psi-psi_s))^2

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
