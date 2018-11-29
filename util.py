import warnings
import inspect
import numpy as np


def check_bounds(bounds, shape):
    lb = np.full(shape, -np.inf)
    ub = np.full(shape, np.inf)
    if bounds is None:
        return (lb, ub)

    assert type(bounds) == tuple and len(
        bounds) == 2, "bounds must be a 2-tuple"

    if bounds[0] is not None:
        lb = np.atleast_1d(bounds[0])
        if lb.size == 1:
            lb = np.full(shape, lb.ravel()[0], dtype=float)
        else:
            assert lb.shape == shape, "invalid shape for bounds"

    if bounds[1] is not None:
        ub = np.atleast_1d(bounds[1])
        if ub.size == 1:
            ub = np.full(shape, ub.ravel()[0], dtype=float)
        else:
            assert ub.shape == shape, "invalid shape for bounds"

    assert np.all(lb <= ub), "inconsistent bounds"
    return (lb, ub)


def check_unique_string_list(L, n=None, illegal_chars=None):
    if L is None:
        return
    assert type(L) is list, "L must be a list"
    assert np.all([type(s) is str for s in L]), "list must contain strings"
    assert len(L) == len(set(L)), "strings must be unique"
    if n is not None:
        assert len(L) == n, "wrong number of strings"
    if illegal_chars is not None:
        assert np.all([len(set(s) & set(illegal_chars)) == 0 for s in L]), \
            "illegal character"


def copyfields(source_object, destination_object, fieldnames, dowarn=True):
    for field in fieldnames:
        if field not in dir(source_object):
            if dowarn:
                warnings.warn("field not found: {0}".format(field))
            continue
        v = source_object.__getattribute__(field)
        destination_object.__setattr__(field, v)


def copy_if_not_None(x):
    if x is not None:
        x = x.copy()
    return x


def enforce_bounds(x, lb, ub):
    '''
    we sometimes need to explicitly enforce bounds due to some unknown rounding
    error inside the lbfgs-b code. this also copies the input array.
    '''
    x = x.copy()  # probably not necessary due to use of maximum
    x = np.maximum(x, lb)
    x = np.minimum(x, ub)
    return x


def ninputs(f):
    assert callable(f), "f must be callable"
    return len(inspect.signature(f).parameters.keys())


def npositional(f):
    assert callable(f), "f must be callable"
    p = inspect.signature(f).parameters
    return np.sum([p[v].default is inspect._empty for v in p.keys()])


def assign_masked_values(x0, xsub, mask):
    x = x0.copy()
    x[mask] = xsub
    return x


def fix_inputs(x0, f, mask, J=None):
    '''
    fix inputs for which the mask is zero. return the resulting initial
    vector, function and Jacobian
    '''
    x0 = x0.copy()
    mask = mask.copy()
    x0sub = x0[mask]

    def fsub(xsub):
        return f(assign_masked_values(x0, xsub, mask))

    if J is None:
        return (x0sub, fsub)

    def Jsub(xsub):
        Jmatorvec = J(assign_masked_values(x0, xsub, mask))
        if Jmatorvec.ndim > 1:
            return J(assign_masked_values(x0, xsub, mask))[:, mask]
        else:
            return J(assign_masked_values(x0, xsub, mask))[mask]

    return (x0sub, fsub, Jsub)


def roundifint(x):
    ix = int(x)
    if x == ix:
        return ix
    return x


def collapse_columns(X, groups):
    Y = np.zeros((X.shape[0], len(groups)))
    for j, g in enumerate(groups):
        for i in g:
            Y[:, j] += X[:, i]
    return Y


def asvec(x, n):
    if type(x) is list:
        x = np.array(x)
    if type(x) is np.ndarray:
        if x.size != n:
            assert x.size == 1, "invalid size"
            return np.full(n, x[0])
        return x.reshape(-1)
    else:
        return np.full(n, x, dtype=type(x))
