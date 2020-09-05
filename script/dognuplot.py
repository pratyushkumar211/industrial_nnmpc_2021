#!/usr/bin/env python3
"""
Runs gnuplot on a given file.

If something fails in gnuplot, then all output files are deleted so that this
can be used with make. The temporary gnuplot file is always deleted.

Note that since gnuplot is called in the <builddir> directory, any data files
used by the gp file must be specified relative to <builddir>
"""
import sys
import os
import subprocess
import re
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--build", help="build directory (where outputs end up)")
parser.add_argument("--terminal", help="gnpulot terminal to use",
                    default="epslatex")
parser.add_argument("--verbose", help="print status messages",
                    action="store_true")
parser.add_argument("--rc", help="filename to use for rc settings")
parser.add_argument("gpfile", help="gnuplot file to run")

def main(args):
    """Main function."""
    # Get arguments.
    args = vars(parser.parse_args(args))
    builddir = args["build"]
    gpfullfile = os.path.abspath(args["gpfile"])
    gpterm = args["terminal"]
    verbose = args["verbose"]
    gnuplotrc = args["rc"]
    
    # Handle default cases.
    (gpdir, gpfile) = os.path.split(gpfullfile)
    gpfilenoext = os.path.splitext(gpfile)[0]
    gpfiletex = gpfilenoext + ".tex"
    gpfileeps = gpfilenoext + ".eps"
    if builddir is None:
        builddir = gpdir
    gpbuildfile = os.path.join(builddir, gpfile + ".tmpgp")
    
    # Now copy gpfile, replacing terminal.
    if verbose:
        print("Writing temporary gnuplot file <{}>.".format(gpbuildfile))
    with open(gpfullfile, "r") as gpread, open(gpbuildfile, "w") as gpbuild:
        for line in gpread:
            line = re.sub("^#* *set terminal.*",
                          "set terminal {}".format(gpterm), line)
            line = re.sub("^#* *set output.*",
                          "set output '{}'".format(gpfiletex), line)
            gpbuild.write(line)
    
    # Now run gnuplot and see what happens.
    gnuplotargs = ["gnuplot", "--default-settings"]
    if gnuplotrc is not None:
        gnuplotargs += ["-e", "load '{}'".format(gnuplotrc)]
        if verbose:
            print("Using <{}> as rc file.".format(gnuplotrc))
    gnuplotargs += [os.path.relpath(gpbuildfile, builddir)]
    if verbose:
        print("Executing <{}> in directory <{}>.".format(" ".join(gnuplotargs),
                                                         builddir))
    gnuplotprocess = subprocess.Popen(gnuplotargs, cwd=builddir)
    gnuplotprocess.wait()
    
    # Remove temporary gp file and check to make sure gnuplot exited with status 0.
    os.remove(gpbuildfile)
    if gnuplotprocess.returncode != 0:
        os.chdir(builddir)
        if os.path.exists(gpfiletex):
            os.remove(gpfiletex)
        if os.path.exists(gpfileeps):
            os.remove(gpfileeps)
        print("**** Error running gnuplot. Check output.")
    
    # Return status.
    return gnuplotprocess.returncode

if __name__ == "__main__":
    status = main(sys.argv[1:])
    sys.exit(status)
