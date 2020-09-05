"""
io functions for mat files.
"""
import sys
import numpy as np
import scipy.io as sio

def iteritems(d):
    """Returns iteritems of teh given dict."""
    if sys.version_info[0] >= 3:
        ret = d.items
    else:
        ret = d.iteritems
    return ret()

# =============================================================================
# The following three functions from http://stackoverflow.com/questions/7008608
# with some additional modifications.
# =============================================================================
def loadmat(filename, scalararray=False, asdict=True, asrecarray=False, squeeze=True):
    """
    Loads a mat file with sensible  behavior for nested scalar structs.
    
    This function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects.
    
    Optional argument scalararray is a boolean flag: if True, scalar values
    are wrapped in the numpy array data type; if False (default), then they
    are returned as native python floats.
    """
    data = sio.loadmat(filename, struct_as_record=asrecarray,
                       squeeze_me=squeeze)
    if asdict:
        data = _check_keys(data, scalararray, asrecarray)
    return data

def _check_keys(d, scalararray=False, recarray=False):
    """
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    """
    for (key, item) in iteritems(d):
        d[key] = _process_elem(item, scalararray, recarray)
    return d

def _todict(matobj, scalararray, recarray):
    """
    A recursive function which constructs from matobjects nested dictionaries.
    """
    if recarray:
        # More "modern" numpy data type.
        def get(d, k):
            v = d[k]
            if hasattr(v, "shape") and v.shape == ():
                v = v[()]
            return v
        keys = list(matobj.dtype.fields.keys())
    else:
        # Older surrogate of Matlab structs.
        def get(d, k):
            return d.__dict__[k]
        keys = matobj._fieldnames
    d = {}
    for k in keys:
        elem = get(matobj, k)
        d[k] = _process_elem(elem, scalararray, recarray)
    return d
# =============================================================================

def _process_elem(elem, scalararray, recarray):
    """
    Performs processing to go from Matlab to numpy data types.
    
    We split this out to avoid duplication in case additional processing is
    desired.
    """
    # Grab type checking function.    
    check = _is_rec_array if recarray else _is_mat_struct
    
    # First check if element is the "struct" type.
    if check(elem):
        elem = _todict(elem, scalararray, recarray)
    # Now check if element is an array of the "struct" type.
    elif (isinstance(elem, np.ndarray) and elem.size > 0
            and check(elem[(0,)*elem.ndim])):
        for e in np.nditer(elem, op_flags=["readwrite"], flags=["refs_ok"]):
            if np.size(e[()]) > 0:
                e[...] = _process_elem(e[()], scalararray, recarray)
    # If element is a float, wrap it as array if requested.
    elif scalararray and isinstance(elem, float):
        elem = np.array(elem)
    return elem

def _is_mat_struct(x, mat_struct=sio.matlab.mio5_params.mat_struct):
    """
    Returns true if x is of type mat_struct.
    
    Not very Pythonic, but we need this to undo some of the IMO poor choices
    of the Scipy people for loading mat files.
    """
    return isinstance(x, mat_struct)

def _is_rec_array(x):
    """
    Checks if x has a dtype, and if so, whether it has multiple fields.
    
    Although the name suggests otherwise, this isn't actually checking if x is
    a numpy recarray.
    """
    return hasattr(x, "dtype") and x.dtype.fields is not None

# Grab savemat function as well.
def savemat(file_name, mdict, appendmat=False, oned_as="column",
            list_as_cell=True):
    """
    Saves the given dictionary to a mat file.
    
    By default, if any elements of the dictionary are lists, they are saved
    as cell arrays. Set list_as_cell to False to allow scipy to choose what
    to do (which may have unexpected behavior).
    """
    if list_as_cell:
        mdict = _list_to_cell(mdict)
    sio.savemat(file_name, mdict, appendmat=appendmat, oned_as=oned_as)

def _list_to_cell(mdict):
    """Converts all lists in mdict to cell arrays (numpy object arrays)."""
    mdict = mdict.copy()
    for (k, v) in iteritems(mdict):
        if isinstance(v, dict):
            mdict[k] = _list_to_cell(v)
        elif isinstance(v, list):
            mdict[k] = np.array(v, dtype=object)
    return mdict
