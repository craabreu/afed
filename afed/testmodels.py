"""
.. module:: testmodels
   :platform: Unix, Windows
   :synopsis: Adiabatic Free Energy Dynamics Test Model Systems

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html
.. _Topology: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.topology.Topology.html

"""

import afed
import os

from simtk import openmm, unit
from simtk.openmm import app


class TestModel:
    def getPositions(self):
        """
        Gets the positions of all atoms in the model system.

        Returns
        -------
            list(openmm.Vec3)

        """
        return self._pdb.positions

    def getSystem(self):
        """
        Gets the System_ object of this model system.

        Returns
        -------
            openmm.System

        """
        return self._system

    def getTopology(self):
        """
        Gets the Topology_ object of this model system.

        Returns
        -------
            openmm.app.Topology

        """
        return self._pdb.topology


class AlanineDipeptideModel(TestModel):
    """
    A system consisting of a single alanine-dipeptide molecule in a vacuum.

    """
    def __init__(self):
        pdb = self._pdb = app.PDBFile(os.path.join('afed', 'data', 'alanine-dipeptide.pdb'))
        forcefield = app.ForceField('amber10.xml')
        self._system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            removeCMMotion=False,
        )
        atoms = [(a.name, a.residue.name) for a in pdb.topology.atoms()]
        psi_atoms = [('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA'), ('N', 'NME')]
        self._psi_angle = openmm.CustomTorsionForce('theta')
        self._psi_angle.addTorsion(*[atoms.index(i) for i in psi_atoms], [])
        phi_atoms = [('C', 'ACE'), ('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA')]
        self._phi_angle = openmm.CustomTorsionForce('theta')
        self._phi_angle.addTorsion(*[atoms.index(i) for i in phi_atoms], [])
        period = 360*unit.degrees
        self._psi = afed.DrivenCollectiveVariable('psi', self._psi_angle, unit.radians, period)
        self._phi = afed.DrivenCollectiveVariable('psi', self._phi_angle, unit.radians, period)
        value = 180*unit.degrees
        minval = -value
        maxval = value
        T = 1500*unit.kelvin
        mass = 168.0*unit.dalton*(unit.angstroms/unit.radian)**2
        velocity_scale = unit.sqrt(unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*T/mass)
        self._psi_driver = afed.DriverParameter('psi_s', unit.radians, value, T, velocity_scale,
                                                minval, maxval, periodic=True)
        self._phi_driver = afed.DriverParameter('phi_s', unit.radians, value, T, velocity_scale,
                                                minval, maxval, periodic=True)

    def getDihedralAngles(self):
        """
        Gets the Ramachandran dihedral angles :math:`\\psi` and :math:`\\phi` of the alanine
        dipeptide angles.

        Returns
        -------
            psi_angle, phi_angle : openmm.CustomTorsionForce

        """
        return self._psi_angle, self._phi_angle

    def getCollectiveVariables(self):
        """
        Gets driven collective variables concerning the Ramachandran dihedral angles.


        Returns
        -------
            psi, phi : DrivenCollectiveVariable

        """
        return self._psi, self._phi

    def getDriverParameters(self):
        """
        Gets the driver parameters associated to the Ramachandran dihedral angles.

        Returns
        -------
            psi_driver, phi_driver : DrivenCollectiveVariable

        """
        return self._psi_driver, self._phi_driver
