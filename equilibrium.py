import numpy as np
from scipy.optimize import nnls
import warnings
import mathfunc
import functools
import names


def response_polynomial_singlebuffer(beta, Lt, Mt):
    '''
    returns polynomial coefficients to solve for free ligand concentration
    given total concentrations of ligand and macromolecule as well as
    the macroscopic binding constants beta

    highest term is of order nstates, so we have nstates + 1 coefficients.
    For example, if we can bind up to 4 ligands per macromolecule we have 5
    states and a 5-th order polynomial with 6 coefficients.

    we need to solve the equation
    R * Ltot = R * L + Mt * \sum_{i=1}^{nstates-1} i * L^i * beta[i-1]
    for the variable L
    where R is the partition function:
    R = 1 + \sum_{i=1}^{nstates-1} L^i beta[i - 1]

    Lt and Mt can be vectors
    '''
    fromscalar = (type(Lt) != np.ndarray) and (type(Lt) != list)
    Lt = np.atleast_1d(Lt)
    Mt = np.atleast_1d(Mt)

    nstates = beta.size + 1  # number of binding states, including ligand-free.
    nconcentrations = Lt.size

    P = np.zeros((nconcentrations, nstates + 1))  # polynomial coefficients.

    P[:, 1:] += np.append(1.0, beta)  # R * L
    # R * Lt. this is an outer product
    P[:, 0:-1] -= Lt.reshape(nconcentrations, 1) * np.append(1.0, beta)
    # Mtot * \sum_{i=1}^{nstates-1} i * L^i * Kaprod[i-1]
    P[:, 1:-1] += Mt.reshape(nconcentrations, 1) * \
        (beta * np.arange(1, nstates))

    P = P[:, -1::-1]

    if fromscalar:
        P = P[0, :]  # return a flat array if Lt and Mt were not arrays

    return P  # return highest-order terms first


def response_polynomial_multibuffer(beta, Lt, Mt):
    '''
    get response polynomial for a case where a single ligand is bound by
    several macromolecules.

    INPUTS
    beta: list of numpy arrays. beta[i].size is the number of ligands that can
    be bound by the i-th buffer.
    Lt: total concentration of ligand at each step of the experiment. 1D numpy
    array.
    Mt: total concentration of each macromolecule for each step of the
    experiment.
    Mt.shape[0] = Lt.size, the number of steps in the experiment.
    Mt.shape[1] = len(beta), the number of different macromolecules

    Outputs
    # fixme
    '''
    nconcentrations = Lt.size
    nbuffers = Mt.shape[1]

    # binding polynomials
    R = [np.append(1.0, b)[-1::-1] for b in beta]
    a = [r * np.arange(r.size - 1, -1, -1) for r in R]  # saturation = a / R

    Rprod = functools.reduce(np.polymul, R)  # product of binding poylnomials

    # rows of P contain response polynomials with the constant term first

    # contribution of Rprod * Lf, same for each experiment step
    P = np.tile(np.append(Rprod, 0.0), [nconcentrations, 1])

    # contribution of -Rprod * Lt
    P[:, 1:] -= np.outer(Lt, Rprod)

    # contribution of each buffer's term with a factor of Mt
    Mfac, Rprod_others = [], []
    for i in range(0, nbuffers):  # i indexes the macromolecule
        R_others = [r for j, r in enumerate(R) if j != i]
        Rprod_others.append(functools.reduce(np.polymul, R_others))
        Mfac.append(np.polymul(Rprod_others[-1], a[i]))
        P[:, 1:] += np.outer(Mt[:, i], Mfac[-1])

    return P, Rprod, Mfac, Rprod_others, a


def solve_response_polynomials(P, Lt):

    # free ligand for each concentration of total ligand:
    Lf = np.zeros(Lt.size, dtype=float)
    nz = Lt != 0
    Lf[nz] = mathfunc.unique_real_roots(P[nz, :],
                                        lb=np.zeros(nz.sum()),
                                        ub=Lt[nz])
    return Lf


def Lt2Lf_2mvbuffers(Ka1, Ka2, Mt1, Mt2, Lt):
    '''
    Solve for free ligand in the presence of two buffers that bind one ligand
    each. This could be two independent binding sites on the same molecule, in
    which case Mt1 == Mt2.

    Some ideas from "Single-experiment displacement assay for quantifying
    high-affinity binding by isothermal titration calorimetry",
    Krainer & Keller 2015.

    See also:
    en.wikipedia.org/wiki/Cubic_function
    #Trigonometric_solution_for_three_real_roots
    '''
    Lf = np.zeros_like(Lt)

    k1 = 1.0 / (Ka1 * Lt)  # dissociation constant 1 in units of Lt
    k2 = 1.0 / (Ka2 * Lt)  # dissociation constant 2 in units of Lt
    m1 = Mt1 / Lt  # macromolecule concentration 1 in units of Lt
    m2 = Mt2 / Lt  # macromolecule concentration 2 in units of Lt

    # we're going to solve for x = (L / Lt ) in the following equation:
    # x^3 + px^2 + qx + r = 0
    p = k1 + k2 + m1 + m2 - 1.0
    q = k1 * (m2 - 1.0) + k2 * (m1 - 1.0) + k1 * k2
    r = -k1 * k2

    v = np.sqrt(p ** 2 - 3.0 * q)
    num = -2.0 * p ** 3 + 9.0 * p * q - 27.0 * r
    denom = 2.0 * v ** 3
    t = np.arccos(num / denom)
    x = (-p + 2.0 * v * np.cos(t / 3.0)) / 3.0

    Lf = x * Lt  # rescale units

    return Lf


def Lt2Lf_multibuffer(beta, Lt, Mt):

    fromscalar = (type(Lt) != np.ndarray) and (type(Lt) != list)
    Lt = np.atleast_1d(Lt).reshape(-1)
    Mt = np.atleast_2d(Mt).reshape(Lt.size, -1)

    P = response_polynomial_multibuffer(beta, Lt, Mt)[0]
    Lf = solve_response_polynomials(P, Lt)

    if fromscalar:
        Lf = Lf[0]
        P = P[0, :]

    return Lf, P


def Lt2Lf_singlebuffer(beta, Lt, Mt):

    fromscalar = (type(Lt) != np.ndarray) and (type(Lt) != list)
    Lt = np.atleast_1d(Lt).reshape(-1)
    Mt = np.atleast_1d(Mt).reshape(-1)

    P = np.atleast_2d(response_polynomial_singlebuffer(beta, Lt, Mt))
    Lf = solve_response_polynomials(P, Lt)

    if fromscalar:
        Lf = Lf[0]
        P = P[0, :]

    return Lf, P


def Lfgradients_multibuffer(beta_all, Lt, Mt, Lfpowers, Rterms, R, s):
    '''
    beta_all is a list of 1D np arrays

    Rterms is a list of 2D np arrays

    Lt is 1D np array

    s, Mt and R are a 2D np arrays (s is saturation)
    '''
    BC = buffering_capacity_multibuffer(Mt, Rterms, R, Lfpowers[:, 1],
                                        s, beta_all)

    dLf_dLt = 1.0 / BC  # BC = dLt_dLf

    dLf_dMt = -s / BC.reshape(-1, 1)

    dLf_dbeta = []
    for j, beta in enumerate(beta_all):
        # partial derivatives:
        ds_dbeta = dsaturation_dbeta(beta, Lfpowers, R[:, j], s[:, j])

        dLf_db = ds_dbeta * (-Mt[:, j] / BC).reshape(-1, 1)
        dLf_dbeta.append(dLf_db)

    return dLf_dLt, dLf_dMt, dLf_dbeta


def dsaturation_dbeta(beta, Lfpowers, R, s):
    '''
    partial derivative of buffer saturation w.r.t. beta with free ligand
    concentration held fixed. rows are conditions, columns are beta elements.

    s is saturation
    R is value of the binding polynomial
    '''
    nstates = Lfpowers.shape[1]
    ii = np.arange(1, nstates)
    return np.add.outer(-s, ii) * Lfpowers[:, 1:] / R.reshape(-1, 1)


def buffering_capacity_multibuffer(Mt, Rterms, R, Lf, saturation, beta):
    '''
    calculates total buffering capacity, given each buffer's beta vector and
    total concentration in each conditions and total ligand in each condition.

    total buffering capacity is defined as the derivative of total ligand with
    respect to free ligand.

    Rterms is a list of numpy arrays with
    Rterms[k][j, i] = beta[k][i] * Lf[j] ** i
    '''
    BC = 1.0
    for k, rt in enumerate(Rterms):
        BC += dsaturation_dLf(Lf, rt, R[:, k],
                              saturation[:, k], beta[k]) * Mt[:, k]

    return BC


def dsaturation_dLf(Lf, Rterms, Rval, s, beta):
    '''
    s is saturation
    '''
    nstates = Rterms.shape[1]  # number of binding states, including ligand-free
    z = Lf == 0
    nz = ~z
    d = np.empty(Lf.shape)
    d[nz] = (np.dot(Rterms[nz, :], np.arange(0, nstates) ** 2) / Rval[nz] -
             s[nz] ** 2) / Lf[nz]
    d[z] = beta[0]
    return d


def Lf2bsfracs(beta, Lf):
    '''
    calculate binding state fractions from free ligand concentration and
    macroscopic binding constants beta
    '''
    Lfpowers = Lf.reshape(-1, 1) ** np.arange(beta.size + 1)
    Rterms = Lfpowers * np.append(1.0, beta)
    R = Rterms.sum(axis=1)
    F = Rterms / R.reshape(-1, 1)
    return F, Lfpowers, Rterms, R


def binding_equilibrium(beta, Lt, Mt):
    assert type(beta[0]) is np.ndarray, "beta must be a list of numpy arrays"
    nbuffers = len(beta)
    nb = [b.size for b in beta]  # number of ligands each macromolecule binds
    nbmax = np.max(nb)
    nsteps = Lt.size

    Lf, P = Lt2Lf_multibuffer(beta, Lt, Mt)

    Lfpowers = np.power.outer(Lf, np.arange(0, nbmax + 1))

    R = np.zeros((nsteps, nbuffers))
    bsfracs = np.zeros((nbuffers, nsteps, nbmax + 1), dtype=float)
    Rterms = []
    for i in range(0, nbuffers):
        ns = nb[i] + 1  # number of binding states for this buffer
        Rterms.append(Lfpowers[:, :ns] * np.append(1.0, beta[i]))
        R[:,  i] = Rterms[-1].sum(axis=1)
        bsfracs[i, :, :ns] = Rterms[-1] / R[:, i:i+1]

    return bsfracs, P, Lfpowers, R, Rterms


def dF_dbeta_singlebuffer(beta, dLf_dP, dF_dLf, Lfpowers, R, F, Lt, Mt):
    '''
    derivative of binding state fractions w.r.t beta for fixed totals
    concentrations of a single ligand and macromolecule. requires Lfpowers,
    so the response poylnomial has already been solved before this function is
    called.
    '''
    m = Lt.size
    n = beta.size  # max. number of bound ligands
    '''
    for a given choice of Lt and Mt, dLf_dbeta consists of a sum of two vectors
    a and b that left-multiplied by dLf_dP:
    a) dPi_dbetai, through which the i-th beta contributes to the ith
    coefficient in P, and
    b) a shifted vector of ones through which the i-th beta contributes to the
    i+1-th coefficient in P. thereby, beta[0] influences P[:, -3] and beta[-1]
    influences P[:, 0]
    '''
    dPi_dbetai = Mt.reshape(-1, 1) * np.arange(1, n + 1) - Lt.reshape(-1, 1)
    '''
    note that dPi_dbetai[0] is the rate of increase of the first order (linear)
    term of P w.r.t the first beta, beta[0]

    also note that the ordering of dLf_dbeta is the same as the ordering of
    beta, despite the fact that P has the opposite ordering.
    '''
    dLf_dbeta = dLf_dP[:, -3::-1] + dLf_dP[:, -2:0:-1] * dPi_dbetai

    # partial derivatives of F with Lf held fixed:
    dF_dbeta_partial = dbsfracs_dbeta(beta, Lfpowers, R, F)
    '''
    J[i, :, :] will store the derivative of the binding state fractions F
    w.r.t. beta[i]. The calculation includes two terms: the direct partial
    derivative of F w.r.t beta, and a chain-rule term consisting of the partial
    derivative w.r.t free ligand concentration multiplied by the derivative of
    free ligand w.r.t beta
    '''
    J = np.full((n, m, n + 1), np.nan)
    for i in range(n):
        J[i, :, :] = dF_dbeta_partial[i, :, :] + \
            dF_dLf * dLf_dbeta[:, i:i + 1]

    return J


def dF_dLt_singlebuffer(beta, dLf_dP, dF_dLf):
    '''
    derivative of binding state fractions w.r.t. Lt, with Mt and beta fixed.
    '''
    dLf_dLt = -np.dot(dLf_dP[:, -1:0:-1], np.append(1.0, beta))
    return dF_dLf * dLf_dLt.reshape(-1, 1)


def dF_dMt_singlebuffer(beta, dLf_dP, dF_dLf):
    '''
    derivative of binding state fractions w.r.t. Mt, with Lt and beta fixed.
    '''
    dLf_dMt = np.dot(dLf_dP[:, -2:0:-1], beta * np.arange(1, beta.size + 1))
    return dF_dLf * dLf_dMt.reshape(-1, 1)


def regressorfcn_unbuffered_titration(beta,
                                      M0,
                                      L0,
                                      L_other,
                                      V_M,
                                      V_other):
    V = V_M + V_other
    Mt_moles = V_M * M0
    Lt_moles = V_M * L0 + V_other * L_other
    Mt = Mt_moles / V  # total concentration of macromolecule at each step
    Lt = Lt_moles / V  # total concentration of ligand at each step

    Lf, P = Lt2Lf_singlebuffer(beta, Lt, Mt)
    F, Lfpowers, _, R = Lf2bsfracs(beta, Lf)

    bscons = F * Mt.reshape(-1, 1)  # binding state concentrations

    model_state = dict(bscons=bscons,
                       P=P,
                       R=R,
                       F=F,
                       Mt=Mt,
                       Lt=Lt,
                       Lfpowers=Lfpowers)

    return bscons, model_state


def regressorjac_unbuffered_titration(beta,
                                      M0,
                                      L0,
                                      L_other,
                                      model_state,
                                      V_M,
                                      V_other):
    V = V_M + V_other
    R, P = model_state['R'], model_state['P']  # binding, response polynomials
    Vf_M = V_M / V  # volume fraction of solution containing macromolecule
    Lt, Mt, F = model_state['Lt'], model_state['Mt'], model_state['F']
    Lfpowers = model_state['Lfpowers']
    m = Lt.size

    # get partial derivatives of macromolecule binding state fractions w.r.t
    # free ligand concentration
    dF_dLf = dbsfracs_dLf(beta, Lfpowers, R)
    dLf_dP = mathfunc.droot_dpolynomial(P, Lfpowers[:, 1])

    dbscons_dbeta = dF_dbeta_singlebuffer(beta,
                                          dLf_dP,
                                          dF_dLf,
                                          Lfpowers,
                                          R,
                                          F,
                                          Lt,
                                          Mt) * Mt.reshape(-1, 1)

    # derivatives w.r.t. base macromolecule concentration. We use the chain
    # rule and the fact that Mt = M0 * Vf_M
    dF_dM0 = dF_dMt_singlebuffer(beta,
                                 dLf_dP,
                                 dF_dLf) * Vf_M.reshape(-1, 1)
    # bscons = F * Mt = F * M0 * Vf_M; we use the product rule
    dbscons_dM0 = (dF_dM0 * M0 + F) * Vf_M.reshape(-1, 1)

    # derivatives w.r.t. total ligand with macromolecule
    # we note that Lt = L0 * Vf_M + L_other * Vf_other and use the chain rule
    dF_dL0 = dF_dLt_singlebuffer(beta,
                                 dLf_dP,
                                 dF_dLf) * Vf_M.reshape(-1, 1)
    dbscons_dL0 = dF_dL0 * Mt.reshape(-1, 1)

    return np.concatenate((dbscons_dbeta,
                           dbscons_dM0.reshape(1, m, -1),
                           dbscons_dL0.reshape(1, m, -1)), axis=0)


def regressorfcn_titration_from_Lf(beta, Lf):
    '''
    returns binding state fractions in each column. leftmost is ligand-free.
    free ligand is given, in general this happens in a scenario where buffers
    are used to set the free ligand concentration.
    '''
    # fixme solve roots to use points outside of buffer range?
    F, Lfpowers, _, R = Lf2bsfracs(beta, Lf)
    model_state = {'bsfracs': F, 'R': R, 'Lfpowers': Lfpowers}
    return (F, model_state)


def regressorjac_titration_from_Lf(beta, model_state):
    '''
    returns derivatives of binding state fractions with respect to Adair-Klotz
    coefficients beta
    J.shape = (nbindingsteps, nconcentrations, nbindingsteps + 1)
    '''
    return dbsfracs_dbeta(
        beta,
        model_state['Lfpowers'],
        model_state['R'],
        model_state['bsfracs']
    )


def dbsfracs_dLf(beta, Lfpowers, R):
    '''
    derivatives of binding state fractions w.r.t free ligand concentration.
    in the case of known concentrations of total ligand and macromolecule, this
    should be interpreted as a partial derivative as Lf may depend on beta etc.
    '''

    m = Lfpowers.shape[0]  # number of different free ligand concentrations
    n = beta.size  # number of binding steps. there are n + 1 binding states.

    b = np.hstack((1.0, beta))
    d = np.full((m, n), np.nan)

    # calculate derivative of binding polynomial w.r.t. free ligand
    dR_dx = np.dot(Lfpowers[:, 0:-1], beta * np.arange(1, n + 1))

    dxpow_dx = np.hstack((
        np.zeros((m, 1)),
        Lfpowers[:, 0:-1] * np.arange(1, n + 1)
    ))

    d = b * (
        dxpow_dx * R.reshape(-1, 1) -
        Lfpowers * dR_dx.reshape(-1, 1)
    ) / (R ** 2).reshape(-1, 1)

    return d


def dbsfracs_dbeta(beta, Lfpow, R, F):
    '''
    this function computes the partial derivative of binding state fractions
    with respect to the macroscopic binding coefficients beta, with free ligand
    held fixed.
    Lf is a vector of free ligand concentrations (not an input).
    Lfpow is a matrix whos columns are powers of Lf. first power is zero.
    R is the binding polynomial. R = np.dot(Lfpow, np.hstack(1.0, beta))
    F is a matrix of binding state fractions
    '''
    m = Lfpow.shape[0]  # number of different free ligand concentrations
    n = beta.size  # number of binding steps. there are n + 1 binding states.

    d = np.full((n, m, n + 1), np.nan)
    for j in range(0, n):
        z = -F
        z[:, j + 1] += 1.0
        d[j, :, :] = z * (Lfpow[:, j + 1] / R).reshape(-1, 1)

    return d


def beta2gamma(beta):
    '''
    given a set of binding constants beta, determine the free ligand
    concentrations gamma for which each binding step is half completed.
    works by root finding.
    '''
    if beta.size == 0:
        return beta.copy()
    assert np.all(beta >= 0), "betas cannot be negative"
    assert beta[-1] > 0, "cannot calculate gammas when last beta is zero"
    n = beta.size  # number of binding sites
    R = np.hstack((1.0, beta))  # binding polynomial, lower terms first
    ii = np.arange(0, n + 1)  # order of binding poylnomial terms
    gamma = np.full(n, np.nan)
    for i in range(0, n):
        Q = R * ((-1.0) ** (ii <= i))
        g = np.roots(Q[-1::-1])
        ok = (g > 0) & (np.imag(g) == 0)
        assert ok.sum() == 1, "failed to find unique positive real root"
        gamma[i] = np.real(g[ok][0])
    return gamma


def gamma2beta(gamma, tol=1e-10):
    '''
    give free ligand concentrations gamma at which each binding step is half
    completed, find the binding constants beta.
    works by solving a linear system.
    '''
    if gamma.size == 0:
        return gamma.copy()
    assert np.all(gamma > 0), "gammas must be strictly positive"
    assert np.all(np.diff(gamma) >= 0), "gammas must be increasing"
    n = gamma.size  # number of binding sites
    A = gamma_Amat(gamma)
    b = np.ones(n)
    beta = np.linalg.lstsq(A, b)[0]
    if np.any(beta < 0):
        beta = nnls(A, b)[0]
        e = np.abs(gamma - beta2gamma(beta)).max()
        if e > tol:
            warnings.warn(
                "inconsistent betas/gammas, max divergence: {0}".format(e))
    return beta


def gamma_Amat(gamma):
    n = gamma.size  # number of binding sites
    A = gamma.reshape(-1, 1) ** np.arange(1, n + 1).reshape(1, -1)
    for i in range(0, n):
        A[i, 0:i] *= -1.0
    return A


def dbeta_dgamma(gamma):
    n = gamma.size
    beta = gamma2beta(gamma)
    db_dg = np.empty((n, n))
    A = gamma_Amat(gamma)

    # each row of dA_dgamma is a derivative w/respect to one gamma
    dA_dgamma = np.arange(1, n + 1) * A / gamma.reshape(-1, 1)
    for i in range(n):
        v = np.zeros(n)
        v[i] = -np.dot(dA_dgamma[i, :], beta)
        db_dg[:, i] = np.linalg.lstsq(A, v)[0]

    return db_dg


def dbeta_dgamma0_and_deltagamma(gamma):
    n = gamma.size
    B = (np.arange(n).reshape(-1, 1) >=
         np.arange(n).reshape(1, -1)).astype('float')
    return np.dot(dbeta_dgamma(gamma), B)


def guess_gamma(nbindingsteps, Lt, Mt,
                maxgamma=np.inf, mingamma=1e-3, maxdgamma=np.inf,
                maxgamma0=np.inf):
    '''
    try to initialize gamma such that all binding steps will happen in a
    generally plausible range of free ligand concentrations given total ligand
    and macromolecule concentrations
    '''
    assert mingamma < maxgamma, "mingamma must be strictly less than maxgamma"
    maxLfmin = (Lt - Mt * nbindingsteps).max()
    if maxLfmin > 0 and Lt.size > 1:
        gmax = np.minimum(maxLfmin * 1.25, maxgamma)
        gmin = np.maximum(gmax * 0.001, mingamma)
    else:
        gmax = np.minimum(0.06 * Mt.max(), maxgamma)
        gmin = np.maximum(0.01 * gmax, mingamma)
    gmin = np.minimum(gmin, maxgamma0)
    if nbindingsteps > 1:  # conditional avoids nans
        gmax = np.minimum(gmax, gmin + 0.6 * (nbindingsteps - 1) * maxdgamma)
        gamma = np.linspace(gmin, gmax, nbindingsteps)
    else:
        gamma = np.full(1, (gmin + gmax) / 2.0)
    return gamma


def guess_gamma_fromLf(nbindingsteps, Lf, maxgamma=np.inf, maxdgamma=np.inf):
    if Lf.size == 1:
        gamma = Lf.copy()
    else:
        gmax = np.minimum(Lf.max(), maxgamma)
        if gmax == 0:
            gmax = maxgamma / 2.0
        gmin = np.minimum(Lf.min(), 0.25 * maxgamma)
        if gmin == 0:
            gmin = gmax / 4.0
        gmax = np.minimum(gmax, gmin + 0.6 * (nbindingsteps - 1) * maxdgamma)
        gamma = np.linspace(gmin, gmax, nbindingsteps)
    return gamma


def iseqmodel(model):
    for name in model.parameter_names:
        if name.startswith('k_{-') or name.startswith('k_{+') or \
           name.startswith('\\tau_{'):
            return False
    for name in model.parameter_names:
        if name.startswith('\\beta') or name.startswith('\\gamma'):
            return True
    return False


def getbeta(m):
    isbeta = np.array([names.isbetaname(name) for name in m.parameter_names])
    beta_index = np.flatnonzero(isbeta)
    beta = m.p[beta_index]
    beta_names = [m.parameter_names[i] for i in beta_index]

    return beta, beta_names, beta_index


def getbetas(m):
    basenames_others = []
    pn = m.parameter_names
    index_mm, index_others = np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    fromgammas = np.any([n.startswith('\\gamma') for n in m.parameter_names])
    if fromgammas:
        if '\\gamma_{1}' in pn:
            index_mm = np.append(index_mm, pn.index('\\gamma_{1}'))
            i = 2
            while '\\Delta\\gamma_{{{0}}}'.format(i) in pn:
                j = pn.index('\\Delta\\gamma_{{{0}}}'.format(i))
                index_mm = np.append(index_mm, j)
                i += 1
        gamma_mm = m.p[index_mm].cumsum()
        beta_mm = gamma2beta(gamma_mm)
        prefix = '\\gamma_{'

    else:
        i = 1
        while '\\beta_{{{0}}}'.format(i) in pn:
            j = pn.index('\\beta_{{{0}}}'.format(i))
            index_mm = np.append(index_mm, j)
            i += 1
        beta_mm = m.p[index_mm]
        prefix = '\\beta_{'

    basenames = [n[len(prefix):-1]
                 for i, n in enumerate(pn) if n.startswith(prefix)]
    basenames_others = [b for b in basenames if not b.isdigit()]
    index_others = np.array([pn.index(prefix + b + '}')
                             for b in basenames_others], dtype=int)
    if fromgammas:
        gamma_others = m.p[index_others]
        beta_others = 1.0 / gamma_others
    else:
        beta_others = m.p[index_others]

    return beta_mm, beta_others, basenames_others, index_mm, index_others


def getgammas(m):
    basenames_others = []
    pn = m.parameter_names
    index_mm, index_others = np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    fromgammas = np.any([n.startswith('\\gamma') for n in m.parameter_names])
    if fromgammas:
        if '\\gamma_{1}' in pn:
            index_mm = np.append(index_mm, pn.index('\\gamma_{1}'))
            i = 2
            while '\\Delta\\gamma_{{{0}}}'.format(i) in pn:
                j = pn.index('\\Delta\\gamma_{{{0}}}'.format(i))
                index_mm = np.append(index_mm, j)
                i += 1
        gamma_mm = m.p[index_mm].cumsum()
        prefix = '\\gamma_{'

    else:
        i = 1
        while '\\beta_{{{0}}}'.format(i) in pn:
            j = pn.index('\\beta_{{{0}}}'.format(i))
            index_mm = np.append(index_mm, j)
            i += 1
        beta_mm = m.p[index_mm]
        gamma_mm = beta2gamma(beta_mm)
        prefix = '\\beta_{'

    basenames = [n[len(prefix):-1]
                 for i, n in enumerate(pn) if n.startswith(prefix)]
    basenames_others = [b for b in basenames if not b.isdigit()]
    index_others = np.array([pn.index(prefix + b + '}')
                             for b in basenames_others], dtype=int)
    if fromgammas:
        gamma_others = m.p[index_others]
    else:
        beta_others = m.p[index_others]
        gamma_others = 1.0 / beta_others

    return gamma_mm, gamma_others, basenames_others, index_mm, index_others


def get_beta_names(models):
    # note that the reduction results will be sorted
    beta_names = np.array(functools.reduce(np.union1d,
                                           [getbeta(m)[1] for m in models]))
    subnames = [bn[len('\\beta_{'):-1] for bn in beta_names]
    '''
    identify betas that describe sequential binding of macromolecule
    '''
    ismm = np.array([sn.isdigit() for sn in subnames])
    index_mm = np.flatnonzero(ismm)
    index_others = np.flatnonzero(~ismm)
    beta_names_mm = beta_names[index_mm]
    beta_names_others = beta_names[index_others]
    return beta_names_mm, beta_names_others


def assign_gammas(m, gamma_mm, gamma_others, basenames_others):
    p = m.p.copy()
    pn = m.parameter_names
    if np.any([n.startswith('\\gamma') for n in pn]):
        if gamma_mm.size > 0 and '\\gamma_{1}' in pn:
            p[pn.index('\\gamma_{1}')] = gamma_mm[0]
            for i in range(1, gamma_mm.size):
                j = pn.index('\\Delta\\gamma_{{{0}}}'.format(i + 1))
                p[j] = gamma_mm[i] - gamma_mm[i - 1]
        for i, bn in enumerate(basenames_others):
            gname = '\\gamma_{{{0}}}'.format(bn)
            if gname in pn:
                j = pn.index(gname)
                p[j] = gamma_others[i]
    elif np.any([n.startswith('\\beta') for n in m.parameter_names]):
        beta_mm = gamma2beta(gamma_mm)
        beta_others = 1.0 / gamma_others
        if beta_mm.size > 0 and '\\beta_{1}' in pn:
            for i in range(0, beta_mm.size):
                j = pn.index('\\beta_{{{0}}}'.format(i + 1))
                p[j] = beta_mm[i]
        for i, bn in enumerate(basenames_others):
            bname = '\\beta_{{{0}}}'.format(bn)
            if bname in pn:
                j = pn.index(bname)
                p[j] = beta_others[i]
    m.set_parameters(p)


def average_gammas(models):
    gamma_mm = np.zeros(0)
    n_mm = np.zeros(0)
    gamma_others = np.zeros(0)
    n_others = np.zeros(0)
    basenames_others = []
    for m in models:
        g, go, bso, _, _ = getgammas(m)
        if g.size > 0:
            if gamma_mm.size == 0:
                gamma_mm = g
                n_mm = np.ones_like(g)
            else:
                gamma_mm += g
                n_mm += 1
        for i, name in enumerate(bso):
            if name in basenames_others:
                j = basenames_others.index(name)
                gamma_others[j] += go[i]
                n_others[j] += 1
            else:
                basenames_others.append(name)
                gamma_others = np.append(gamma_others, go[i])
                n_others = np.append(n_others, 1.0)

    gamma_mm /= n_mm
    gamma_others /= n_others

    for m in models:
        assign_gammas(m, gamma_mm, gamma_others, basenames_others)
