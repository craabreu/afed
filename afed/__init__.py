"""
AFED for OpenMM
Adiabatic Free Energy Dynamics with OpenMM
"""


from ._version import get_versions
from .afed import *  # noqa: F401, F403
from .integrators import *  # noqa: F401, F403

# Handle versioneer:
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
