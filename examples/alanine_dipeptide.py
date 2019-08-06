import afed
import argparse

from simtk import openmm, unit
from sys import stdout

parser = argparse.ArgumentParser()
parser.add_argument('--platform', dest='platform', help='the computation platform', default='CUDA')
parser.add_argument('--water', dest='water', help='the water model', default=None)
args = parser.parse_args()

timestep = 3*unit.femtoseconds
temp = 300*unit.kelvin
tau = 10*unit.femtosecond

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
    # print(force.__class__.__name__,force.getForceGroup())

# Define AFED integrator:
integrator = afed.MassiveMiddleNHCIntegrator(
    temp,
    tau,
    timestep,
    # 0.25*unit.femtosecond,
    model.getDrivingForce(),
    respaLoops=respa_loops,
    # parameterLoops=6,
)
print(integrator)

simulation = openmm.app.Simulation(topology, system, integrator, platform, properties)
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
