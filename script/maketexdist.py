#!/usr/bin/env python3
"""
Make a distributable .zip file for a .tex source.

The .zip file includes all files that are used by the root .tex source. Users
should be able to compile the document using only the files in the .zip and
standard LaTeX packages.

Some mild cleaning is performed on the source file, in particular removing any
\\graphicspath commands and adjusting \\bibliography commands if an xbib
file is provided.
"""
import sys
import os
import argparse
import subprocess
import re
import zipfile

parser = argparse.ArgumentParser(add_help=False, epilog=__doc__)
parser.add_argument("--help", action="help", help="print help")
parser.add_argument("--include-texmf-home", action="store_true",
                    help="include files in user's texmf home directory")
parser.add_argument("--xbib",
                    help="extracted .bib file to use in bibliographies")
parser.add_argument("--output", help="name for output .zip file")
parser.add_argument("texfile", help=".tex file to distribute")

def getincludedfiles(texbase, texmfhome=False, bib=False):
    """Returns a list of included files by reading a .fls file."""
    # Grab all raw filenames.
    rawfiles = set()
    with open(texbase + ".fls", "r") as fls:
        for line in fls:
            (prefix, file) = line.strip().split(" ", maxsplit=1)
            if prefix == "INPUT":
                rawfiles.add(file)
    
    # Choose which files to keep.
    texdir = os.path.dirname(texbase)
    keepexts = {".sty", ".cls", ".tex", ".pdf"}
    if bib:
        keepexts.add(".bib")
    if texmfhome:
        texmfhome = kpsewhich("--var-val", "TEXMFHOME")
    else:
        texmfhome = None
    files = []
    for file in rawfiles:
        (_, fileext) = os.path.splitext(file)
        if fileext in keepexts:
            filedir = os.path.dirname(file)
            if filedir == "" or filedir.startswith("."):
                files.append(os.path.join(texdir, file))
            elif texmfhome is not None:
                if file.startswith(texmfhome):
                    files.append(file)
    
    # Do not include base .tex file.
    texbase = texbase + ".tex"
    if texbase in files:
        files.remove(texbase)
    return files


def kpsewhich(*args):
    """Runs kpsewhich and returns output."""
    kpsewhich = subprocess.run(["kpsewhich"] + list(args),
                               stdout=subprocess.PIPE)
    output = kpsewhich.stdout.decode().strip()
    return output


def maketexdist(texfile, output=None, include_texmf_home=False, xbib=None):
    """Makes a distributable .zip file for a .tex source."""
    (texbase, ext) = os.path.splitext(texfile)
    if ext != ".tex":
        raise ValueError("File '{}' is not a .tex file!".format(texfile))
    if output is None:
        output = texbase + ".zip"
    
    # Get files to copy and write to .zip.
    includedfiles = getincludedfiles(texbase, texmfhome=include_texmf_home)
    with zipfile.ZipFile(output, "w") as dist:
        # Standard included files.
        for file in includedfiles:
            filebase = os.path.basename(file)
            dist.write(file, filebase)
        
        # xbib file.
        if xbib is not None:
            xbibname = os.path.splitext(os.path.basename(xbib))[0] + ".bib"
            dist.write(xbib, xbibname)
        
        # Cleaned .tex file.
        dist.writestr(os.path.basename(texfile),
                      "".join(cleantexfile(texfile, xbib=xbib)))
    

def cleantexfile(texfile, xbib=None):
    """Cleans a source .tex file for distribution. Returns a generator."""
    if xbib is not None:
        (xbib, _) = os.path.splitext(os.path.split(xbib)[1])
    with open(texfile, "r") as tex:
        for line in tex:
            sline = line.lstrip()
            if sline.startswith("\\graphicspath"):
                line = ""
            if xbib is not None:
                line = re.sub(r"(\\bibliography|\\addbibresource)\{.*\}",
                              r"\1{%s}" % xbib, line)
            yield line


def main(args):
    """Runs main function."""
    args = vars(parser.parse_args(args))
    maketexdist(**args)


if __name__ == "__main__":
    main(sys.argv[1:])
