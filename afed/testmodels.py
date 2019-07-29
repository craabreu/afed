"""
.. module:: testmodels
   :platform: Unix, Windows
   :synopsis: Adiabatic Free Energy Dynamics Test Model Systems

.. moduleauthor:: Charlles Abreu <abreu@eq.ufrj.br>

.. _System: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.openmm.System.html
.. _Topology: http://docs.openmm.org/latest/api-python/generated/simtk.openmm.app.topology.Topology.html

"""

import os

from simtk import openmm
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
        forcefield = app.ForceField('charmm36.xml')
        self._system = forcefield.CreateSystem(
            pdb.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=None,
            removeCMMotion=False,
        )
        atoms = [(a.name, a.residue.name) for a in pdb.topology.atoms()]
        psi_atoms = [('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA'), ('N', 'NME')]
        self._psi = openmm.CustomTorsionForce('theta')
        self._psi.addTorsion(*[atoms.index(i) for i in psi_atoms], [])
        phi_atoms = [('C', 'ACE'), ('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA')]
        self._phi = openmm.CustomTorsionForce('theta')
        self._phi.addTorsion(*[atoms.index(i) for i in phi_atoms], [])

    def getDihedralAngles(self):
        """
        Gets the Ramachandran dihedral angles :math:`\\psi` and :math:`\\phi` of the alanine
        dipeptide angles.

        Returns
        -------
            phi, psi : openmm.CustomTorsionForce

        """
        return self._psi, self._phi
