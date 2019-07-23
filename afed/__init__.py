"""
AFED for OpenMM
Adiabatic Free Energy Dynamics with OpenMM
"""

# Add imports here
from .afed import *

# Handle versioneer
from ._version import get_versions
versions = get_versions()
__version__ = versions['version']
__git_revision__ = versions['full-revisionid']
del get_versions, versions
