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
        outputs : list
            A list of valid outputs (file names and/or output streams).

    """

    def __init__(self, files):
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


class ExtendedStateDataReporter(app.StateDataReporter):
    """
    An extension of OpenMM's StateDataReporter_ class, which outputs information about a simulation,
    such as energy and temperature, to a file.

    All original functionalities of StateDataReporter_ are preserved.

    Besides, it is possible to report the current values of all collective variables and driver
    parameters associated to a passed `~afed.afed.DrivingForce` object.

    Keyword Args
    ------------
        drivingForce : DrivingForce, default=None
            A :class:`~afed.afed.DrivingForce` whose associated collective variables and driver parameters
            will be reported.

    """

    def __init__(self, file, reportInterval, **kwargs):
        self._drivingForce = kwargs.pop('drivingForce', None)
        super().__init__(file, reportInterval, **kwargs)
        self._backSteps = -sum([self._speed, self._elapsedTime, self._remainingTime])

    def _add_item(self, lst, item):
        if self._backSteps == 0:
            lst.append(item)
        else:
            lst.insert(self._backSteps, item)

    def _constructHeaders(self):
        headers = super()._constructHeaders()
        if self._drivingForce is not None:
            for cv in self._drivingForce._driven_variables:
                self._add_item(headers, f'{cv._name} ({cv._dimension})')
            for parameter in self._drivingForce._driver_parameters:
                self._add_item(headers, f'{parameter._name} ({parameter._dimension})')
        return headers

    def _constructReportValues(self, simulation, state):
        values = super()._constructReportValues(simulation, state)
        if self._drivingForce is not None:
            for cv in self._drivingForce.getCollectiveVariableValues(simulation.context):
                self._add_item(values, cv)
            for parameter in self._drivingForce._driver_parameters:
                self._add_item(values, simulation.context.getParameter(parameter._name))
        return values
