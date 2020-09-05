"""Returns variables and dependencies from a Makefile as JSON."""
import sys
import argparse
import tempfile
import subprocess
import json


parser = argparse.ArgumentParser(description=__doc__, add_help=False)
parser.add_argument("--help", action="help",
                    help="display help")
parser.add_argument("--output", help="file for output (otherwise stdout)")
parser.add_argument("makefile", help="name of Makefile to process")


def main(args):
    """Parses arguments and runs main functions."""
    kwargs = vars(parser.parse_args(args))
    makedb = getmakedatabase(kwargs["makefile"])
    outputfile = (sys.stdout if kwargs["output"] is None
                  else open(kwargs["output"], "w"))
    with outputfile:
        json.dump(makedb, outputfile, indent=4)


def getmakedatabase(makefile="Makefile"):
    """Returns a dict with the Makefile database."""
    with tempfile.TemporaryFile() as makedboutput:
        # Get Make database and discard header.
        make = subprocess.Popen(["make", "-pq", "--file", makefile],
                                stdout=makedboutput)
        make.wait()
        makedb = StrFileIterator(makedboutput,
                                 name="{} database".format(makefile))
        for _ in untilstartswith(makedb, "# Variables"):
            pass
        
        # Get user-defined Make variables.
        variables = {}
        for line in untilstartswith(makedb, "# Files"):
            if line.startswith("# makefile (from"):
                # Check next line.
                varline = next(makedb)
                if varline.startswith("define "):
                    var = varline[len("define "):].strip()
                    val = "".join(untilstartswith(makedb, "endef"))
                elif "=" in varline:
                    (var, val) = varline.split("=", maxsplit=1)
                    var = var.strip(": ")
                    val = val.strip()
                else:
                    var = None
                if var is not None:
                    variables[var] = val
        
        # Get file dependencies.
        dependencies = {}
        for line in untilstartswith(makedb, "# files hash-table stats:"):
            if line == "\n":
                continue
            elif line.startswith("# Not a target:"):
                next(makedb) # Throw away the next line.
            elif not any(map(line.startswith, ["#", "\t"])):
                # Need to check next line.
                filestatus = next(makedb)
                if not filestatus.startswith("#  Phony target"):
                    (file, deps) = line.split(":")
                    deps = [d for d in deps.strip().split(" ") if len(d) > 0]
                    dependencies[file.strip()] = deps
    
    # Return combined dictionary.
    return dict(variables=variables, dependencies=dependencies)


def untilstartswith(file, start, error=True):
    """
    Yields from file until the given value startswith start.
    
    If error is true, raises an error if iterator is exhausted before start
    is encountered. Otherwise, iteration just ends.
    """
    while True:
        try:
            line = next(file)
        except StopIteration:
            if error:
                raise IOError("%s ended before %r was encountered!"
                              % (file.name, start))
            else:
                break
        if line.startswith(start):
            break
        yield line


# Prior to Python 3.6, all output of subprocess.Popen jobs was returned as
# bytes. We use the following wrapper class so that we can dump the output to
# a file as binary data but then read back as strings.
class StrFileIterator:
    def __init__(self, file, encoding="utf8", name=None):
        """Returns an iterator to the lines of file as strings."""
        file.seek(0)
        self.__file = file
        self.name = name if name is not None else file.name
        self.encoding = encoding
        
    def __next__(self):
        """Returns the next line in the file as a string."""
        return next(self.__file).decode(self.encoding)
