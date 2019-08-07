import afed
import argparse

from simtk import openmm, unit
from sys import stdout

waters = ['spce', 'tip3p', 'tip4pew', 'tip5p']
parser = argparse.ArgumentParser()
parser.add_argument('--platform', dest='platform', help='the computation platform', default='CUDA')
parser.add_argument('--water', dest='water', help='the water model', choices=waters, default=None)
args = parser.parse_args()

timestep = 6*unit.femtoseconds
temp = 300*unit.kelvin
tau = 10*unit.femtosecond
psi_atoms = [('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA'), ('N', 'NME')]
phi_atoms = [('C', 'ACE'), ('N', 'ALA'), ('CA', 'ALA'), ('C', 'ALA')]

platform = openmm.Platform.getPlatformByName(args.platform)
properties = dict(Precision='mixed') if args.platform == 'CUDA' else dict()

model = afed.AlanineDipeptideModel(water=args.water)
system, topology, positions = model.getSystem(), model.getTopology(), model.getPositions()

# Split forces into multiple time scales:
respa_loops = [4, 6, 1]  # time steps = 0.125 fs, 0.5 fs, 3 fs
for force in system.getForces():
    if isinstance(force, openmm.NonbondedForce):
        force.setForceGroup(2)
        force.setReciprocalSpaceForceGroup(2)
    elif isinstance(force, openmm.PeriodicTorsionForce):
        force.setForceGroup(1)

# Add driven collective variables (dihedral angles):
atoms = [(a.name, a.residue.name) for a in topology.atoms()]
psi = openmm.CustomTorsionForce('theta')
psi.addTorsion(*[atoms.index(i) for i in psi_atoms], [])
psi = afed.DrivenCollectiveVariable('psi', psi, unit.radians, period=360*unit.degrees)

phi = openmm.CustomTorsionForce('theta')
phi.addTorsion(*[atoms.index(i) for i in phi_atoms], [])
phi = afed.DrivenCollectiveVariable('phi', phi, unit.radians, period=360*unit.degrees)

# Add driver parameters [Ref: Chen et al., JCP 137 (2), art. 024102, 2012]:
T_dihedrals = 1500*unit.kelvin
mass_dihedrals = 168.0*unit.dalton*(unit.angstroms/unit.radian)**2
K_dihedrals = 2.78E3*unit.kilocalories_per_mole/unit.radians**2
velocity_scale = unit.sqrt(unit.BOLTZMANN_CONSTANT_kB*unit.AVOGADRO_CONSTANT_NA*T_dihedrals/mass_dihedrals)

psi_driver = afed.DriverParameter('psi_s', unit.radians, psi.evaluate(positions), T_dihedrals,
                                  velocity_scale, -180*unit.degrees, 180*unit.degrees, periodic=True)
phi_driver = afed.DriverParameter('phi_s', unit.radians, phi.evaluate(positions), T_dihedrals,
                                  velocity_scale, -180*unit.degrees, 180*unit.degrees, periodic=True)

# Add driving force:
dihedrals = afed.HarmonicDrivingForce()
dihedrals.addPair(psi, psi_driver, K_dihedrals)
dihedrals.addPair(phi, phi_driver, K_dihedrals)
dihedrals.setForceGroup(0)  # parameter velocity integration at fastest time scale
system.addForce(dihedrals)

# Define AFED integrator:
integrator = afed.MassiveMiddleNHCIntegrator(
    temp,
    tau,
    timestep,
    dihedrals,
    respaLoops=respa_loops,
    parameterLoops=6,
)
print(integrator)

simulation = openmm.app.Simulation(topology, system, integrator)
simulation.context.setPositions(positions)
simulation.minimizeEnergy()
simulation.context.setVelocitiesToTemperature(temp)

data_reporter = afed.StateDataReporter(
    afed.MultipleFiles(stdout, 'alanine_dipeptide.csv'),
    10,
    afedIntegrator=integrator,
    step=True,
    potentialEnergy=True,
    temperature=True,
    collectiveVariables=True,
    driverParameters=True,
    parameterTemperatures=True,
    speed=True,
)

pdb_reporter = openmm.app.PDBReporter(
    'alanine_dipeptide.pdb',
    100,
)

simulation.reporters += [data_reporter, pdb_reporter]
simulation.step(100000)
