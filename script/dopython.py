"""
Runs a python script in the given directory.

Note that since python is called in a directory, any paths inside the script
must be relative to that directory.

Note that this script can be invoked using either Python 2 or 3.
"""

import sys
import os
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.normpath(thisdir + "/../lib"))

import custompath
custompath.add()

import argparse
parser = argparse.ArgumentParser(description=__doc__)

parser.add_argument("--dir", help="directory in which script is run")
parser.add_argument("--path", help="folder to add to Python's path",
                    action="append")
parser.add_argument("pyfile", help="python script to run",
                    type=os.path.abspath)

# Get execfile for Python 3.
if sys.version_info[0] >= 3:
    import runpy
    execfile = runpy.run_path

def runscript(script, rundir=None, pathadds=None):
    """Runs the given script in the given directory."""
    # Do path things.
    if pathadds is not None:
        for p in pathadds:
            if p not in sys.path:
                sys.path.append(p)
                
    # Choose directory.
    if rundir is None:
        rundir = thisdir
    
    # Attempt to run script.
    os.chdir(rundir)
    return execfile(script)

if __name__ == "__main__":
    args = vars(parser.parse_args(sys.argv[1:]))
    runscript(args["pyfile"], args["dir"], args["path"])
