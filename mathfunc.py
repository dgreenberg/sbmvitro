import numpy as np
from scipy.optimize._numdiff import approx_derivative

extobj = np.linalg.linalg.get_linalg_error_extobj(
        np.linalg.linalg._raise_linalgerror_eigenvalues_nonconvergence)


def droot_dpolynomial(p, x):
    '''
    for a root x of a polynomial p, finds the derivative of x with respect to
    the coefficients of p.
    '''
    singleinput = p.ndim == 1

    p = np.atleast_2d(p)
    m = p.shape[0]
    n = p.shape[1] - 1  # polynomial degree
    assert n > 0, "polynomial must be of degree > 1"

    x = np.atleast_1d(x).reshape(m, 1)
    xpowers = x ** np.arange(n, -1, -1)

    # polynomial coefficients for dp_dx
    pprime = p[:, :-1] * np.arange(n, 0, -1)
    dp_dx = np.sum(pprime * xpowers[:, 1:], axis=1)
    dx_dp = -xpowers / dp_dx.reshape(-1, 1)

    if singleinput:
        dx_dp = dx_dp.reshape(-1)

    return dx_dp


def droot_dpolynomial_fd(p, x, rel_step=None):
    '''
    finite differences version for testing purposes
    '''
    def nearest_root(p):
        rts = np.roots(p)
        if np.imag(x) == 0:
            # keep root real if it was previously so (good idea?)
            rts = np.real(rts[(np.imag(rts) == 0)])
        if rts.size == 0:
            return np.nan
        return rts[np.argmin(np.abs(rts - x))]

    return approx_derivative(nearest_root, p, rel_step=rel_step)


def unique_real_roots(P, lb=None, ub=None, eps=1e-6):
    n = P.shape[0]
    if lb is None:
        lb = np.full(n, -np.inf)
    if ub is None:
        ub = np.full(n, np.inf)

    eps_lb = np.maximum(np.abs(lb), eps) * eps
    eps_ub = np.maximum(np.abs(ub), eps) * eps

    u = np.zeros(n)
    A = np.diag(np.ones(P.shape[1] - 2), -1)  # pre-allocate

    for i in range(0, n):
        rts = quickroots(P[i, :], A)
        ok = np.flatnonzero(
            (np.isreal(rts)) &
            (rts >= lb[i] - eps_lb[i]) &
            (rts <= ub[i] + eps_ub[i])
            )
        assert ok.size == 1, \
            "could not find a unique real root within the bounds" \
            + "\nrts\t\t=\t" + str(rts[0]) + " " + str(rts[1]) \
            + "\nok\t\t=\t" \
            + str((np.isreal(rts))) \
            + " " + str((rts >= lb[i])) \
            + " " + str((rts <= ub[i])) \
            + '\nlb[i]\t=\t'+str(lb[i]) \
            + '\nub[i]\t=\t'+str(ub[i])

        u[i] = rts[ok[0]].real

    return u


def quickroots(p, A):
    '''
    roots function that skips input and error checking. use with care.
    requires pre-allocated array A for eigenvalue calculation.
    '''
    if p[0] == 0:
        return np.roots(p)
    A[0, :] = -p[1:] / p[0]

    return np.linalg.linalg._umath_linalg.eigvals(A,
                                                  signature='d->D',
                                                  extobj=extobj)
