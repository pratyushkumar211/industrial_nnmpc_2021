# Help message generated automatically. Run from command line with --help
# option to see all available options.
"""
Runs one or more python files options. These files should produce pdf plots
using matplotlib.

If something fails in python, half-completed plots may be left behind. Thus,
if you are using this with make, you should write your python scripts so that
plots are saved at the very end.

Supplying a --grey option sets a flag to indicate greyscale plots. However, it
is up to the individual python scripts to check whether plottools.GREYSCALE
is True and change colors accordingly.

Note that since python is called in the BUILD directory (if provided), any
data files used by the py file must be specified relative to BUILD.
"""
import sys
if sys.version_info[0] >= 3:
    import runpy
    execfile = runpy.run_path

import os
thisdir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.normpath(thisdir + "/../lib"))

import custompath
custompath.add()

import argparse
import traceback

# Helper function for parser.
def getfilecheck(ext=None, directory=False, exists=True, toAbs=True):
    """
    Returns a funcion to check whether inputs are valid files/directories.
    """
    def filecheck(s):
        s = str(s)
        if toAbs:
            try:
                s = os.path.abspath(s)
            except: # Catch everything here, although presumably this can't fail.
                raise argparse.ArgumentTypeError("unable to get absolute path")
        if ext is not None and not s.endswith(ext):
            raise argparse.ArgumentTypeError("must have '%s' extension" % (ext,))
        if exists:
            if directory:
                if not os.path.isdir(s):
                    raise argparse.ArgumentTypeError("must be an existing directory")
            elif not os.path.isfile(s):
                raise argparse.ArgumentTypeError("must be an existing file")
        return s
    return filecheck

parser = argparse.ArgumentParser(add_help=False, description=
    "runs matplotlib on one or more files with various options",
    epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--help", help="print this help",action="help")
parser.add_argument("--build", help="directory in which to build",
                    type=getfilecheck(directory=True))
parser.add_argument("--path", help="folder to add to Python path",
                    action="append", default=[])
parser.add_argument("--grey", help="set greyscale flag", action="store_true")
parser.add_argument("--font", help="pick font style",
                    choices=["paper","presentation","poster"], default="paper")
parser.add_argument("--backend", help="specify Matplotlib backend to use",
                    default="default")
parser.add_argument("pyfile",help="python source file",action="append",
                    type=getfilecheck(ext=".py"), default=[])

# Parse command line arguments.
options = vars(parser.parse_args(sys.argv[1:]))
pyfullfiles = options["pyfile"]
builddir = options["build"]
pathadds = options["path"]
greyscale = options["grey"]
font = options["font"]
backend = options["backend"]

# Import matplotlib and choose backend.
import matplotlib
if backend != "default":
    matplotlib.use(backend)
import matplotlib.pyplot as plt
try:
    import plottools
except ImportError:
    print("*** Warning: unable to import plottools. Some functionality may "
          "be unavailable.")
    plottools = None

# Adjust path.
for p in pathadds:
    if p not in sys.path:
        sys.path.append(p)
    
# Figure out directories.
thisdir = os.getcwd()
if builddir is None:
    builddir = thisdir

usePresentationFont = False
usePosterFont = False
sansmathnumbers = False
if plottools is not None:
    plottools.GREYSCALE = greyscale
    if font == "presentation":
        usePresentationFont = True
        fontfamily = "sans-serif"
        sansmathnumbers = True
    elif font == "poster":
        usePosterFont = True
        fontfamily = "sans-serif"
        sansmathnumbers = True
    else:
        fontfamily = "serif"
    plottools._init(fontfamily, presentation=usePresentationFont,
                    times=usePosterFont, poster=usePosterFont,
                    sansmathnumbers=sansmathnumbers)

# Change to build directory and run python on each file.
exitstatus = 0
plt.ioff()
plt.show = lambda *args : None
if len(builddir) > 0:    
    os.chdir(builddir)
for pyfullfile in pyfullfiles:
    try:
        execfile(pyfullfile)
    except: # Catch all errors.
        traceback.print_exc()
        print("***Error running python on %s" % os.path.basename(pyfullfile))    
        exitstatus += 1
    finally:
        os.chdir(thisdir)        
sys.exit(exitstatus)
