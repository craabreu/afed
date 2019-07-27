"""
.. module:: output
   :platform: Unix, Windows
   :synopsis: Adiabatic Free Energy Dynamics Outputs

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _CustomCVForce: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.CustomCVForce.html
.. _StateDataReporter: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.StateDataReporter.html

"""

import sys

from simtk.openmm import app


class MultipleFiles:
    """
    Allows the use of multiple outputs in an OpenMM Reporter.

    Parameters
    ----------
        A list of valid outputs (file names and/or output streams).

    """

    def __init__(self, *files):
        self._files = list()
        for output in files:
            self._files.append(open(output, 'w') if isinstance(output, str) else output)

    def __del__(self):
        for output in self._files:
            if output != sys.stdout and output != sys.stderr:
                output.close()

    def write(self, message):
        for output in self._files:
            output.write(message)

    def flush(self):
        for output in self._files:
            output.flush()


class StateDataReporter(app.StateDataReporter):
    """
    An extension of OpenMM's StateDataReporter_ class, which outputs information about a simulation,
    such as energy, temperature, etc.

    All original functionalities of StateDataReporter_ are preserved.

    Besides, it is possible to report the current values of all collective variables, driver
    parameters, and other properties associated to a passed AFED integrator.

    Parameters
    ----------
        afedIntegrator : afed.integrators, default=None
            An object of any class defined in :mod:`afed.integrators`.

    Keyword Args
    ------------
        collectiveVariables : bool, defaul=False
            Whether to output the values of the collective variables included in the integrator's
            :class:`~afed.afed.DrivingForce`.
        DriverParameters : bool, defaul=False
            Whether to output the values of the driver parameters included in the integrator's
            :class:`~afed.afed.DrivingForce`.

    """

    def __init__(self, file, reportInterval, afedIntegrator, **kwargs):
        self._afed_integrator = afedIntegrator
        self._collective_variables = kwargs.pop('collectiveVariables', False)
        self._driver_parameters = kwargs.pop('DriverParameters', False)
        super().__init__(file, reportInterval, **kwargs)
        self._backSteps = -sum([self._speed, self._elapsedTime, self._remainingTime])

    def _add_item(self, lst, item):
        if self._backSteps == 0:
            lst.append(item)
        else:
            lst.insert(self._backSteps, item)

    def _constructHeaders(self):
        headers = super()._constructHeaders()
        if self._afed_integrator is not None:
            driving_force = self._afed_integrator._driving_force
            if self._collective_variables:
                for cv in driving_force._driven_variables:
                    self._add_item(headers, f'{cv._name} ({cv._dimension})')
            if self._driver_parameters:
                for parameter in driving_force._driver_parameters:
                    self._add_item(headers, f'{parameter._name} ({parameter._dimension})')
        return headers

    def _constructReportValues(self, simulation, state):
        values = super()._constructReportValues(simulation, state)
        if self._afed_integrator is not None:
            driving_force = self._afed_integrator._driving_force
            if self._collective_variables:
                for cv in driving_force.getCollectiveVariableValues(simulation.context):
                    self._add_item(values, cv)
            if self._driver_parameters:
                for parameter in driving_force._driver_parameters:
                    self._add_item(values, simulation.context.getParameter(parameter._name))
        return values
