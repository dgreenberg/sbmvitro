import numpy as np
import warnings


def sbrate(y, dy_dt, n, b, u, kon_B, koff_B):
    '''
    rate equation for sequential binding
    vector ordering: y = [[M], [L_1M], ..., [L_nM], [L],
        [B_1], [B_2], ... [B_{nB}],
        [LB_1], [LB_2], ... [LB_{nB}]]^T
    returns dy/dt
    '''
    # FIXME: remove states [B], [L_nM], add params [B]_tot, [M]_tot
    c = y[0:n + 1]
    L = y[n + 1]
    B = y[n + 2]
    LB = y[n + 3]

    # rate for each reaction
    r = b * L * c[0:n] - u * c[1:n + 1]  # macromolecule binding steps
    rB = kon_B * L * B - koff_B * LB     # each buffer's single binding step

    dy_dt[0] = -r[0]
    for i in range(1, n):
        dy_dt[i] = r[i - 1] - r[i]
    dy_dt[n] = r[n - 1]
    dy_dt[n + 1] = -r.sum() - rB
    dy_dt[n + 2] = -rB
    dy_dt[n + 3] = rB

    return dy_dt


def sbrate_jac(y, J, n, b, u, kon_B, koff_B):
    '''
    Jacobian of rate equation derivative function sbrate
    with respect to state vector y
    '''
    iL = n + 1  # index into y for [L]
    iB = n + 2
    iLB = n + 3

    c = y[0:n + 1]           # binding state concentrations
    L = y[iL]                # ligand concentration
    B = y[n + 2]  # ligand-free buffer concentrations
    # note that we don't need LB for the calculation of J

    # derivative of f w.r.t. ligand-free state concentration
    J[0, 0] = -b[0] * L  # d^2[M]  / dt d[M]
    J[1, 0] = b[0] * L  # d^2[LM] / dt d[M]
    J[iL, 0] = -b[0] * L  # d^2[L]  / dt d[M]

    # derivatives of f w.r.t. partially saturated state concentrations
    for ii in range(1, n):
        # d^2[L_{ii-1}M] / dt d[L_{ii}M]
        J[ii - 1, ii] = u[ii - 1]
        J[ii,     ii] = -u[ii - 1] - b[ii] * \
            L   # d^2[L_{ii}M]   / dt d[L_{ii}M]
        # d^2[L_{ii+1}M] / dt d[L_{ii}M]
        J[ii + 1, ii] = b[ii] * L
        # d^2[L]         / dt d[L_{ii}M]
        J[iL,     ii] = -b[ii] * L + u[ii - 1]

    # derivative of f w.r.t. ligand-saturated state concentration
    J[n - 1, n] = u[n - 1]   # d^2[L_{n-1}M] / dt d[L_nM]
    J[n,     n] = -u[n - 1]  # d^2[L_nM]     / dt d[L_nM]
    J[iL,    n] = u[ii - 1]  # d^2[L]        / dt d[L_nM]

    # derivative of f w.r.t. L
    # d^2[M]       / dt d[L]
    J[0, iL] = -b[0] * c[0]
    for ii in range(1, n):
        J[ii, iL] = b[ii - 1] * c[ii - 1] - \
            b[ii] * c[ii]   # d^2[L_{ii}M] / dt d[L]
    # d^2[L_nM]    / dt d[L]
    J[n,  iL] = b[n - 1] * c[n - 1]
    J[iL, iL] = -np.dot(b, c[0:n]) - kon_B * B  # d^2[L] / dt d[L]

    J[iB,  iL] = -kon_B * B  # d^2[B]  / dt d[L]
    J[iLB, iL] = kon_B * B  # d^2[LB] / dt d[L]

    # derivative of f w.r.t ligand-free and ligand-bound buffer concentrations
    J[iB,   iB] = -kon_B * L  # d^2[B]  / dt d[B]
    J[iLB,  iB] = kon_B * L   # d^2[LB] / dt d[B]
    J[iB,  iLB] = koff_B      # d^2[B]  / dt d[LB]
    J[iLB, iLB] = -koff_B     # d^2[LB] / dt d[LB]
    J[iL,   iB] = -kon_B * L  # d^2[L]    / dt d[B]
    J[iL,  iLB] = koff_B      # d^2[L]    / dt d[LB]

    return J

# keep a copy of the python functions for debugging etc.
sbrate_python = sbrate
sbrate_jac_python = sbrate_jac

try:
    # use numba to compile these functions if available
    from numba import jit
    sbrate = jit('f8[:](f8[:], f8[:], int64, f8[:], f8[:], f8, f8)',
                 nopython=True, nogil=True
                 )(sbrate)
    sbrate_jac = jit('f8[:,:](f8[:], f8[:, :], int64, f8[:], f8[:], f8, f8)',
                     nopython=True, nogil=True
                     )(sbrate_jac)
except ImportError:
    warnings.warn("numba is not available, using slower python kinetics")
