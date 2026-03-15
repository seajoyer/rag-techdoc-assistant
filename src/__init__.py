"""
Root package init — exposes the main subpackage
"""

from . import data_acquisition, chunking

__all__ = [
    "data_acquisition",
    "chunking"
]
