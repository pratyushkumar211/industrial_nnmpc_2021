# This script must be able to run on both Python 2 and 3!
"""
Sets up symbolic links and checks Python packages. Run via `make install`.

By default, any files/links that are already present are left alone. However,
if you use `make forceinstall`, they will all be removed.
"""

import collections
import os
import shutil
import sys
import traceback
from distutils.version import StrictVersion as Version

LinkDir = collections.namedtuple("LinkDir",["name","hg","there","here","version"])
PyPkg = collections.namedtuple("PyPkg",["name","hg","version"])
LinkStatus = collections.namedtuple("LinkStatus",["link","path","version"])
INSTALLSTATUS = 0

# Check a command line option.
FORCE = ("--force" in sys.argv)

# ************************************
# Edit links and Python packages here.
# ************************************

# Syntax: LinkDir(other repo base name, hg location, name of folder in repo, name for link, min version)
# Use None as the min version if you don't wish to do any version checking.
# By default, the list doesn't include anything. Commented out is an example.
linkdirs = [
#    LinkDir("HighLevel","https://bitbucket.org/risbeck/highlevel",".","lib/HighLevel","1.3"),
]

# Syntax: PyPkg(package name, hg location, min version)
# By default, this list includes plottools.
pypkgs = [
    PyPkg("plottools", "https://bitbucket.org/risbeck/plottools", "1.4.0"),
]
# *********************

# First take care of repos.
helpstr = """
This repo uses files from

    %s

Attempting to create symbolic links.
""" % "\n    ".join("%s : %s" % (l.name, l.hg) for l in linkdirs)
if len(linkdirs) > 0:
    print(helpstr)

# Define some helper functions.
def topline():
    print("\n" + r"\/ "*20)
def botline():
    print(r"/\ "*20 + "\n")
def getrepoversion(name, path, start="VERSION="):
    """Looks for repo version string."""
    version = None
    try:
        versionfilename = os.path.join(path, ".__%s__" % name)
    except:
        import pdb; pdb.set_trace()
    try:
        with open(versionfilename, "r") as versionfile:
            for line in versionfile:
                if line.startswith(start):
                    version = line[len(start):].strip()
                    break
    except (OSError, IOError):
        pass
    return version
    
# Now loop through directories and try to make links.
links = collections.defaultdict(lambda : [])
for l in linkdirs:
    status = "good"
    path = None
    
    # If force option is specified, try to remove it.     
    if FORCE and os.path.exists(l.here):
        try:
            if os.path.isdir(l.here) and not os.path.islink(l.here):
                shutil.rmtree(l.here)
            else:
                os.remove(l.here)
        except OSError:
            status = "bad"
    
    # Check if the link already exists. Otherwise, guess location.
    if os.path.islink(l.here) and status == "good": 
        status = "found"
        path = l.here
    elif status == "good":
        absdir = os.path.abspath(os.path.join("..", l.name, l.there))
        if os.path.isdir(absdir):
            try:
                os.symlink(absdir, l.here)
                path = absdir
            except OSError:
                status = "bad"
        else:
            status = "bad"
    
    # Check version.
    version = None
    if status in ["good", "found"] and l.version is not None:
        version = getrepoversion(l.name, path)
        if version is None or Version(version) < Version(l.version):
            status = "old"
    
    # Now add link to appropriate place.
    links[status].append(LinkStatus(l, path, version))

# Finally, go back through links and tell user what happened.
INSTALLSTATUS = 0
if len(links["good"]) > 0:
    print("The following symlinks were created automatically:")
    for (l, ab, version) in links["good"]:
        print("    %s -> %s (Version %s)" % (l.here, ab, version))
    print("")
if len(links["old"]) > 0:
    topline()
    print("The following links were found but were out of date:")
    for (l, path, version) in links["old"]:
        print("    %s : %s (have %s, need %s)" % (l.name, path, version,
            l.version))
    print("Please update these repos.")
    botline()    
    INSTALLSTATUS = 1
if len(links["bad"]) > 0:
    topline()
    print("Please make the following symlinks manually:\n")
    for (l, _, _) in links["bad"]:
        print("    %s -> %s : %s" % (l.here, l.hg, l.there))
    print("\nPaths are relative to the repo root directory.\n")    
    botline()
    INSTALLSTATUS = 1
if len(links["found"]) > 0:
    print("The following links were found:")
    for (l, path, version) in links["found"]:
        print("    %s : %s (Version %s)" % (l.name, path, version))
    print ("Please verify that they are up to date, or use --force to"
        " overwrite them.")

# Now ready for custom python packages.
helpstr = """
This repo uses custom python packages

    %s
    
These packages exist somewhere on your python path or in the lib folder of
this repo.

Attempting to import packages.
""" % "\n    ".join(p.name for p in pypkgs)
if len(pypkgs) > 0:
    print(helpstr)

# Now loop and save bad packages.
thisdir = os.path.dirname(os.path.abspath(__file__))
libdir = os.path.normpath(os.path.join(thisdir, "../lib"))
sys.path.insert(0, libdir) # Put lib at beginning of path.
pkgs = collections.defaultdict(lambda : [])
errmessages = []
for p in pypkgs:
    status = "good"    
    path = None
    version = None
    try:
        thispkg = __import__(p.name)
        path = thispkg.__file__
    except ImportError:
        status = "bad"
    except: # Catch everybody else.
        errmessages.append(
            ("**** UNKNOWN ERROR IMPORTING %s ****\n" % (p.name,))
            + traceback.format_exc())
        status = "bad"
    else:
        if path.startswith(libdir):
            path = path[len(libdir) - 3:] # Show relative path if in lib.
        if p.version is not None:
            version = getattr(thispkg, "__version__", None)
            if version is None or Version(version) < Version(p.version):
                status = "old"
    pkgs[status].append(LinkStatus(p, path, version))

# Let user know what happened.
if len(pkgs["good"]) > 0:
    print("The following packages were found automatically:\n")
    for (p, ab, ver) in pkgs["good"]:
        print("    %s -> %s (Version %s)" % (p.name, ab, ver))
    print("")
if len(pkgs["bad"]) > 0 or len(errmessages) > 0:
    topline()
    if len(pkgs["bad"]) > 0:
        print ("\nPlease download missing packages from the following links "
            "and follow installation instructions.")
        for (p, _, _) in pkgs["bad"]:
            print("    %s: %s" % (p.name, p.hg))
        print("\n")
    if len(errmessages) > 0:
        print("The following unknown errors ocurred while importing packages.")
        print("\n\n".join(errmessages))
    botline()
    INSTALLSTATUS += 2
if len(pkgs["old"]) > 0:
    topline()
    print("The following packages were found but were out of date:")
    for (l, path, version) in pkgs["old"]:
        print("    %s : %s (have %s, need %s)" % (l.name, path, version,
            l.version))
    print("Please update these packages.")
    botline()    
    INSTALLSTATUS += 2

# All done.
if INSTALLSTATUS == 0:
    print("\nEverything successful.\n")
else:
    topline()
    print("Errors during install. Please follow instructions above.")
    botline()
sys.exit(INSTALLSTATUS)
