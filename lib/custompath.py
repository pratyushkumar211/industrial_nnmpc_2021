"""
Adds important locations to Python's path.

Paths can be defined relatively in this script, and they will be added
absolutely so that, e.g., you can import files from these directories.
"""

import sys
import os
thisdir = os.path.dirname(os.path.abspath(__file__))

# User pick folders here.
PATHS = [
    thisdir + "/../lib",
    thisdir + "/../script",
    thisdir + "/../build",
]

def addToPath(addPaths, atend=False):
    """Adds the given paths to beginning (end if atend=True) of sys.path."""    
    adds = []
    for p in addPaths:
        fullp = os.path.abspath(p)
        if fullp in sys.path:
            sys.path.pop(sys.path.index(fullp))
        adds.append(fullp)
    if atend:
        sys.path = sys.path + adds
    else:
        sys.path = adds + sys.path

def add(extrapaths=None):
    """Adds default locations to beginning of path."""
    if extrapaths is None:
        extrapaths = []            
    addToPath(PATHS + [thisdir] + extrapaths)

# An extra function to search multiple directories for a given .mat file.
def safeload(f, altDirs=None):
    if altDirs is None:
        altDirs = ["build"]
    try:
        import plottools.matio
    except ImportError:
        raise ImportError("plottools not found. Try `make install`.")
    altDirs = ["."] + altDirs
    for d in altDirs:
        try:
            ans = plottools.matio.loadmat(os.path.join(d, f))
            break
        except IOError:
            pass
    else:
        raise IOError("Unable to open <{}> from any directories!".format(f))
    return ans

