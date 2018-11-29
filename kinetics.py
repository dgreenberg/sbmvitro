'''modeling of kinetics and stopped flow fluorescence experiments'''
import copy
import warnings
import numpy as np
import equilibrium as eq
import modelfit as mf
import names
import constants
import util
from rates import sbrate, sbrate_jac  # causes compilation if numba installed
from scipy.integrate import ode


def stoppedflow_params(exp,
                       nbindingsteps,
                       mingamma=10e-9,  # Molar
                       mindgamma=2.5e-9,  # Molar
                       initial_contamination_macromolecule=0.0,
                       max_contamination_buffer=0.1,
                       max_contamination_mm=np.inf,
                       concentration_unit=1e-6,
                       min_purity_mm=0.5,
                       max_purity_mm=3.0,
                       min_purity_buffer=0.85,
                       max_purity_buffer=1.0,
                       dead_time_uncertainty_ms=0.5,
                       max_affinity_error_buffer=0.5,  # vs. constants.py
                       max_tauobs_error_buffer=0.5,  # vs. constants.py
                       maxgamma=np.inf,  # max mm gamma, init. only
                       maxgamma0=np.inf,  # actual bound
                       maxdgamma=np.inf,
                       gammaparam=True,
                       buffer_purity=True,
                       buffer_contamination=True,
                       macromolecule_purity=True,
                       macromolecule_contamination=True,
                       tau0=0.25,
                       minkoff=0.05,
                       maxkoff=1e4,
                       mintau=1e-4,  # seconds
                       maxtau=20.0,  # seconds
                       maxkon=1e10,  # sec^-1*M^-1. standard diffusion limit
                       **kwargs_unused):
    '''
    get initial values, bounds and names for standard titration parameters.
    also returns extra_inputs dict for regressorfcn and regressorjac
    '''
    assert exp['type'] == 'stoppedflow', "wrong exp type"

    hasbuffer = 'buffer' in exp.keys() and \
        exp['buffer'] != 'none'
    mm, ll = exp['macromolecule'], exp['ligand']
    if 'protein_purification' in exp.keys():
        pp = exp['protein_purification']
    else:
        pp = None

    cu = concentration_unit
    maxgamma = np.minimum(maxgamma, maxgamma0 + nbindingsteps * maxdgamma)
    if not np.isinf(maxgamma):
        nfac = 100.0 * nbindingsteps
        mindgamma = np.minimum(mindgamma, (maxgamma - mingamma) / nfac)

    if tau0 >= maxtau:
        tau0 = 0.9 * maxtau

    Lt_nominal = exp['total ligand'] / concentration_unit
    Mt_nominal = exp['total macromolecule'] / concentration_unit

    t = exp['time']
    assert t[0] > 0, "first measurement time must be positive"
    assert np.all(np.diff(t) > 0), "times must be increasing"
    tdiff = np.diff(np.append(0.0, t))
    t -= tdiff / 2.0  # account for averaging of multiple samples

    V = exp['volumes']
    Vf = V / V.sum()

    gamma_mm = eq.guess_gamma(nbindingsteps, Lt_nominal, Mt_nominal,
                              maxgamma=maxgamma / cu, mingamma=mingamma / cu,
                              maxdgamma=maxdgamma / cu,
                              maxgamma0=maxgamma0 / cu)

    beta_mm = eq.gamma2beta(gamma_mm)
    K_a_mm = beta_mm / np.append(1.0, beta_mm[:-1])
    koff_mm, kon_mm = K_atau2koffkon(K_a_mm, tau0)

    if gammaparam:
        tau_mm = np.full(nbindingsteps, tau0)
        p0 = np.hstack((gamma_mm[0], np.diff(gamma_mm), tau_mm))
        lb = np.hstack((mingamma / cu,  # g
                        np.full(nbindingsteps - 1, mindgamma / cu),  # dg
                        np.full(nbindingsteps, mintau)))  # tau
        ub = np.hstack((maxgamma0 / cu,  # gamma
                        np.full(nbindingsteps - 1, maxdgamma / cu),  # dgamma
                        np.full(nbindingsteps, maxtau)))  # tau
        gdg_names, tau_names = names.gdgtnames(nbindingsteps)
        parameter_names = gdg_names + tau_names
    else:
        p0 = np.hstack((koff_mm, kon_mm))
        lb = np.hstack((np.full(nbindingsteps, minkoff),  # koff
                        np.zeros(nbindingsteps)))  # kon
        maxkon_cu = maxkon * concentration_unit
        ub = np.hstack((np.full(nbindingsteps, maxkoff),  # koff
                        np.full(nbindingsteps, maxkon_cu)))  # kon
        koff_names, kon_names = names.rateconstantnames(nbindingsteps)
        parameter_names = koff_names + kon_names

    # dead time
    ddt = dead_time_uncertainty_ms / 1000.0
    tdead0 = exp['dead time (ms)'] / 1000.0
    p0 = np.append(p0, tdead0)
    lb = np.append(lb, np.maximum(5e-4, tdead0 - ddt))  # at least 500 us
    ub = np.append(ub, tdead0 + ddt)
    parameter_names.append('dead time')

    if macromolecule_purity:
        p0 = np.append(p0, 1.0)
        lb = np.append(lb, min_purity_mm)
        ub = np.append(ub, max_purity_mm)
        if pp is not None:
            parameter_names.append('{0} purity ({1})'.format(mm, pp))
        else:
            parameter_names.append('{0} purity'.format(mm))

    if macromolecule_contamination:
        p0 = np.append(p0, initial_contamination_macromolecule)
        lb = np.append(lb, 0.0)
        ub = np.append(ub, max_contamination_mm)
        if pp is not None:
            parameter_names.append('{0} in {1} contamination ratio ({2})'
                                   .format(ll, mm, pp))
        else:
            parameter_names.append('{0} in {1} contamination ratio'
                                   .format(ll, mm))

    if hasbuffer:
        Bt_nominal = exp['total buffer'].reshape(1, 2) / cu
    else:
        Bt_nominal = np.zeros((1, 2))

    extra_inputs = \
        dict(nbindingsteps=nbindingsteps,
             gammaparam=gammaparam,
             t=t,
             Mt_nominal=Mt_nominal,
             Lt_nominal=Lt_nominal,
             Bt_nominal=Bt_nominal,
             Vf=Vf,
             macromolecule_purity=macromolecule_purity,
             buffer_purity=buffer_purity,
             buffer_contamination=buffer_contamination,
             macromolecule_contamination=macromolecule_contamination)

    if not hasbuffer:
        return p0, parameter_names, (lb, ub), extra_inputs

    # fixme buffer batches?
    bb = exp['buffer']
    try:
        # initialize from database
        koff_B, kon_B = constants.rates_sec_M(
            bb,
            pH=exp['pH'],
            T=exp['temperature'],
            IS_mm=exp['ionic strength'] * 1e3)
        kon_B *= cu
        K_a_B = kon_B / koff_B
        gamma_B = 1.0 / K_a_B  # monovalent
        min_gamma_B = gamma_B * (1.0 - max_affinity_error_buffer)
        max_gamma_B = gamma_B * (1.0 + max_affinity_error_buffer)
        assert max_affinity_error_buffer < 1.0, "invalid setting"
        buffer_from_constants = True

    except ValueError:
        warnings.warn("failed to retrieve rate constants for {0}"
                      .format(bb))
        # initialize from guess
        gamma_B = np.array([np.mean(gamma_mm)])
        beta_B = eq.gamma2beta(gamma_B)
        K_a_B = beta_B  # monovalent
        koff_B, kon_B = K_atau2koffkon(K_a_B, tau0)
        min_gamma_B = mindgamma / cu
        max_gamma_B = np.inf
        buffer_from_constants = False

    if gammaparam:
        tau_B = 1.0 / (koff_B * (1.0 + gamma_B * K_a_B))
        if buffer_from_constants:
            min_tau_B = tau_B * (1.0 - max_tauobs_error_buffer)
            max_tau_B = tau_B * (1.0 + max_tauobs_error_buffer)
            assert max_tauobs_error_buffer < 1.0, "invalid setting"
        else:
            min_tau_B, max_tau_B = mintau, maxtau

        p0 = np.hstack((p0, gamma_B, tau_B))
        lb = np.hstack((lb, min_gamma_B, min_tau_B))
        ub = np.hstack((ub, max_gamma_B, max_tau_B))
        parameter_names += ['\\gamma_{{{0}}}'.format(bb),
                            '\\tau_{{{0}}}'.format(bb)]
    else:
        p0 = np.hstack((p0, koff_B, kon_B))
        lb = np.hstack((lb,
                        minkoff,  # koff
                        0.0))  # kon
        ub = np.hstack((ub,
                        maxkoff,  # koff
                        maxkon_cu))  # kon
        parameter_names += ['k_{{-{0}}}'.format(bb),
                            'k_{{+{0}}}'.format(bb)]

    if buffer_purity:
        p0 = np.append(p0, 1.0)
        lb = np.append(lb, min_purity_buffer)
        ub = np.append(ub, max_purity_buffer)
        if 'buffer_batch' in exp.keys():
            parameter_names.append('{0} purity (batch {1})'
                                   .format(bb, exp['buffer_batch']))
        else:
            parameter_names.append('{0} purity (unknown batch)@'.format(bb))

    if buffer_contamination:
        p0 = np.append(p0, 0.0)
        lb = np.append(lb, 0.0)
        ub = np.append(ub, max_contamination_buffer)
        if 'buffer_batch' in exp.keys():
            parameter_names.append('{0} in {1} contamination ratio (batch {2})'
                                   .format(ll, bb, exp['buffer_batch']))
        else:
            parameter_names.append('{0} in {1} contamination '
                                   'ratio (unknown batch)@'.format(ll, bb))

    return p0, parameter_names, (lb, ub), extra_inputs


def stoppedflow_scale_name(exp):
    '''name of data scale for a stopped flow exp'''
    if 'instrument' not in exp.keys():
        return None
    assert '*' not in exp['instrument'], "illegal character"
    scale_name = 'stopped-flow {0}'.format(exp['instrument'])
    if 'emission filter' in exp.keys():
        scale_name += ', em. {0}'.format(exp['emission filter'])
    if 'pmt voltage' in exp.keys():
        v_pmt = int(np.round(exp['pmt voltage']))
        scale_name += ', V_pmt = {0}'.format(v_pmt)
    if 'slit width (nm)' in exp.keys():
        scale_name += ', {0} nm slit width'.format(exp['slit width (nm)'])
    if 'protein_purification' in exp.keys():
        scale_name += ' ({0})'.format(exp['protein_purification'])
        '''logic is that different pools might have different brightnesses
        due to different rates of correct fluorophore formation. this does not
        yet take into account misfolding that could cause changes in binding'''
    return scale_name


def stoppedflow_noise_type(exp):
    '''name of noise type for stoppedflow experiment'''
    if 'instrument' not in exp.keys():
        return None
    # fixme! noise depends on several experimental conditions
    return '{0} noise'.format(exp['instrument'])


def stoppedflow2sbmodel(exp,
                        nbindingsteps,
                        **opts):
    '''create a regressionmodel from a stopped flow exp dict'''
    opts.setdefault('concentration_unit', 1e-6)
    opts.setdefault('mint_stoppedflow', 0.0)
    opts.setdefault('maxF_stoppedflow', np.inf)
    exp = copy.deepcopy(exp)

    p0, parameter_names, param_bounds, extra_inputs = \
        stoppedflow_params(exp, nbindingsteps, **opts)
    regressor_names = names.bindingstatenames(exp, nbindingsteps)
    datatype = 'fluorescence'
    if 'excitation wavelength' in exp.keys():
        wav = util.roundifint(exp['excitation wavelength'])
        datatype = 'fluorescence (\\lambda_{{ex}}={0} nm)'.format(wav)
    else:
        datatype = 'fluorescence (\\lambda_{{ex}}=?)@'

    data = np.mean(exp['data'], axis=1)
    n_averaged = exp['data'].shape[1]

    tmask = np.logical_and(
        extra_inputs['t'] >= opts['mint_stoppedflow'], # too soon after mixing
        data < opts['maxF_stoppedflow'])  # too bright, PMT saturating
    assert tmask.any(), "no time points available after mint_stoppedflow"
    extra_inputs['t'] = extra_inputs['t'][tmask]
    data = data[tmask]

    coef_bounds = (np.zeros((nbindingsteps + 1, 1)),
                   np.full((nbindingsteps + 1, 1), np.inf))

    model = mf.regressionmodel(data,
                               regressorfcn_sf,
                               p0,
                               regressorjac=None,  # use finite differences
                               parameter_names=parameter_names,
                               regressor_names=regressor_names,
                               data_types=[datatype],
                               scale_name=stoppedflow_scale_name(exp),
                               noise_type=stoppedflow_noise_type(exp),
                               param_bounds=param_bounds,
                               coef_bounds=coef_bounds,
                               auto_scale=True,  # for now
                               extra_inputs=extra_inputs)

    model.noise_factor = np.sqrt(n_averaged)
    model.concentration_unit = opts['concentration_unit']
    model.tmask = tmask
    return model


def koffkon2gammatau_bs(koff, kon):
    '''convert from rate constants to K50s and time constants'''
    Ka = np.atleast_1d(kon / koff)
    gamma = eq.beta2gamma(np.atleast_1d(Ka).cumprod())
    kobs = koff * (1.0 + gamma * Ka)
    tau = 1.0 / kobs
    return gamma, tau


def decompose_p(p,
                nbindingsteps,
                gammaparam,
                buffered,
                buffer_purity,
                macromolecule_purity,
                buffer_contamination,
                macromolecule_contamination):

    if gammaparam:
        gamma = p[:nbindingsteps].cumsum()
        tau = p[nbindingsteps:2 * nbindingsteps]
        koff, kon = gammatau_obs2koffkon(gamma, tau)
    else:
        koff = p[:nbindingsteps]
        kon = p[nbindingsteps:2 * nbindingsteps]
    offset = 2 * nbindingsteps

    dead_time = p[offset]
    offset += 1
    if macromolecule_purity:
        purity_mm = p[offset]
        offset += 1
    else:
        purity_mm = 1.0
    if macromolecule_contamination:
        contamination_mm = p[offset]
        offset += 1
    else:
        contamination_mm = 0.0

    koff_B, kon_B, purity_B, contamination_B = 1.0, 1.0, 1.0, 0.0
    if buffered:
        if gammaparam:
            gamma_B = p[offset]  # same as K_d since buffer is monovalent
            offset += 1
            tau_B = p[offset]
            offset += 1
            koff_B = 0.5 / tau_B
            kon_B = koff_B / gamma_B
        else:
            koff_B = p[offset]
            offset += 1
            kon_B = p[offset]
            offset += 1
        if buffer_purity:
            purity_B = p[offset]
            offset += 1
        if buffer_contamination:
            contamination_B = p[offset]
            offset += 1

    return koff, kon, dead_time, purity_mm, contamination_mm, \
        koff_B, kon_B, purity_B, contamination_B


def regressorfcn_sf(p,
                    nbindingsteps=None,
                    gammaparam=None,
                    t=None,
                    Mt_nominal=None,
                    Lt_nominal=None,
                    Bt_nominal=None,
                    Vf=None,
                    macromolecule_purity=None,
                    buffer_purity=None,
                    buffer_contamination=None,
                    macromolecule_contamination=None):
    buffered = Bt_nominal.size > 0
    koff, kon, dead_time, purity_mm, contamination_mm, koff_B, kon_B, \
        purity_B, contamination_B = decompose_p(p,
                                                nbindingsteps,
                                                gammaparam,
                                                buffered,
                                                buffer_purity,
                                                macromolecule_purity,
                                                buffer_contamination,
                                                macromolecule_contamination)

    Mt = Mt_nominal * purity_mm
    Lt = Lt_nominal + Mt_nominal * contamination_mm

    if buffered:
        Bt = Bt_nominal * purity_B
        Lt += Bt_nominal.reshape(-1) * contamination_B

    t_simulation = np.append(0.0, t + dead_time)

    Lf, c_M, B, LB, Lf_premix, bsfracs_premix = \
        stoppedflow_initial_concentrations(koff,
                                           kon,
                                           Mt,
                                           Lt,
                                           Vf,
                                           Bt.reshape(1, -1),
                                           np.array([koff_B]),
                                           np.array([kon_B]))

    y = simulate_kinetics(kon, koff, Lf, c_M, t_simulation,
                          kon_B=kon_B, koff_B=koff_B, B=B, LB=LB).T

    nr = nbindingsteps + 1
    regressors = y[1:, :nr]  # mm binding states only, exclude t=0
    state = dict(LjM=y[:, 0:nr],
                 Lf=y[:, nr],
                 B=y[:, nr + 1],
                 LB=y[:, nr + 2],
                 t=t_simulation,
                 Lf_premix=Lf_premix,
                 bsfracs_premix=bsfracs_premix)

    return regressors, state


def gammatau_obs2koffkon(gamma, tau):
    beta = eq.gamma2beta(np.atleast_1d(gamma))
    b = np.append(1.0, beta)
    Ka = b[1:] / b[:-1]
    kobs = 1.0 / tau
    koff = kobs / (1.0 + gamma * Ka)
    kon = Ka * koff
    return koff, kon


def K_atau2koffkon(K_a, tau):
    n = K_a.size
    tau = np.atleast_1d(tau)
    assert tau.ndim == 1, "tau must be a vector or scalar"
    if tau.size == 1:
        tau = np.full(n, tau[0])
    assert tau.size == n, "tau is incorrectly sized"

    koff = 1.0 / tau

    kon = koff * K_a
    return (koff, kon)


def stoppedflow_initial_concentrations(koff,
                                       kon,
                                       Mt,
                                       Lt,
                                       Vf,
                                       Bt,
                                       koff_B,
                                       kon_B):
    '''
    get initial concentrations of binding states for buffers and macromolecule,
    as well as free ligand, from kinetic parameters and total concentrations in
    each stopped-flow cell
    '''
    nbuffers = koff_B.size
    K_a = kon / koff
    beta_M = K_a.cumprod()
    beta_B = kon_B / koff_B

    # beta_all is a list of numpy arrays
    beta_all = [beta_M] + [beta_B[i:i + 1] for i in range(nbuffers)]

    # get binding states and free ligand in each cell before mixing
    bsfracs, _, Lfpowers, _, _ = \
        eq.binding_equilibrium(beta_all,
                               Lt,
                               np.append(Mt.reshape(2, 1), Bt.T, axis=1))
    Lf = Lfpowers[:, 1]

    # get concentrations after mixing. this is NOT an equilibrium state.
    Lf_mixed = np.dot(Lf, Vf)  # float scalar
    c_M_mixed = np.dot(bsfracs[0, :, :].T, Vf * Mt)  # all binding states
    B_mixed = np.dot(bsfracs[1:, :, 0] * Bt, Vf)  # free
    LB_mixed = np.dot(bsfracs[1:, :, 1] * Bt, Vf)  # bound

    return (Lf_mixed, c_M_mixed, B_mixed, LB_mixed, Lf, bsfracs)


def iskimodel(model):
    for name in model.parameter_names:
        if name.startswith('k_{-') or name.startswith('k_{+') or \
           name.startswith('\\tau_{'):
            return True
    return False


def pki2peq(pki, index_mm, index_others, index_common, gammaparam):
    '''convert kinetic to equilibrium parameters'''
    n_mm = index_mm.size
    n_others = index_others.size
    peq = np.full(pki.size - n_mm - n_others, np.nan)

    # assign parameters that are not gammas or betas in the eq. model
    peq[index_common] = pki[2 * n_mm + 2 * n_others:]

    if gammaparam:  # taus get ignored
        peq[index_mm] = pki[:n_mm]  # gamma / dgamma
        peq[index_others] = pki[2 * n_mm:2 * n_mm + n_others]  # gamma
    else:
        koff_mm = pki[:n_mm]
        kon_mm = pki[n_mm:2 * n_mm]
        koff_others = pki[2 * n_mm:2 * n_mm + n_others]
        kon_others = pki[2 * n_mm + n_others:2 * n_mm + 2 * n_others]

        peq[index_mm] = (kon_mm / koff_mm).cumprod()  # beta_mm
        peq[index_others] = kon_others / koff_others  # beta_others

    return peq


def regressorfcn_eq2ki(pki,
                       index_mm=None,
                       index_others=None,
                       index_common=None,
                       gammaparam=None,
                       regressorfcn_eq=None,
                       regressorjac_eq=None,
                       extra_inputs_eq=None):
    peq = pki2peq(pki, index_mm, index_others, index_common, gammaparam)
    return regressorfcn_eq(peq, **extra_inputs_eq)


def regressorjac_eq2ki(pki,
                       model_state,
                       index_mm=None,
                       index_others=None,
                       index_common=None,
                       gammaparam=None,
                       regressorfcn_eq=None,
                       regressorjac_eq=None,
                       extra_inputs_eq=None):
    peq = pki2peq(pki, index_mm, index_others, index_common, gammaparam)
    dregressors_dpeq = regressorjac_eq(peq, model_state, **extra_inputs_eq)
    ndata = dregressors_dpeq.shape[1]
    nr = dregressors_dpeq.shape[2]  # number of regressors

    nbindingsteps = index_mm.size
    nothers = index_others.size  # number of non-mm buffers

    if gammaparam:
        return np.concatenate((dregressors_dpeq[index_mm, :, :],  # g / dg
                               np.zeros((nbindingsteps, ndata, nr)),  # tau
                               dregressors_dpeq[index_others, :, :],  # g / dg
                               np.zeros((nothers, ndata, nr)),  # tau
                               dregressors_dpeq[index_common, :, :]
                              ), axis=0)

    koff_mm = pki[:nbindingsteps]
    kon_mm = pki[nbindingsteps:2 * nbindingsteps]
    beta_mm = peq[index_mm]
    koff_others = pki[2 * nbindingsteps:2 * nbindingsteps + nothers]
    kon_others = \
        pki[2 * nbindingsteps + nothers:2 * nbindingsteps + 2 * nothers]

    dregressors_dbeta_mm = dregressors_dpeq[index_mm, :, :]
    dregressors_dbeta_others = dregressors_dpeq[index_others, :, :]

    dregressors_dkon_mm = np.empty((nbindingsteps, ndata, nr))
    dregressors_dkoff_mm = np.empty((nbindingsteps, ndata, nr))
    for i in range(nbindingsteps):
        dbeta_dkon = np.zeros(nbindingsteps)
        dbeta_dkon[i:] = beta_mm[i:] / kon_mm[i]
        dbeta_dkoff = np.zeros(nbindingsteps)
        dbeta_dkoff[i:] = -beta_mm[i:] / koff_mm[i]
        dregressors_dkon_mm[i, :, :] = \
            np.sum(dregressors_dbeta_mm *
                   dbeta_dkon.reshape(-1, 1, 1),
                   axis=0)
        dregressors_dkoff_mm[i, :, :] = \
            np.sum(dregressors_dbeta_mm *
                   dbeta_dkoff.reshape(-1, 1, 1),
                   axis=0)

    dregressors_dkon_others = np.empty((nothers, ndata, nr))
    dregressors_dkoff_others = np.empty((nothers, ndata, nr))
    for i in range(nothers):
        # buffers are monovalent so beta = Ka = kon / koff
        dbeta_dkon = 1.0 / koff_others[i]
        dbeta_dkoff = -kon_others[i] / (koff_others[i] ** 2)
        dregressors_dkon_others[i, :, :] = \
            dregressors_dbeta_others[i, :, :] * dbeta_dkon
        dregressors_dkoff_others[i, :, :] = \
            dregressors_dbeta_others[i, :, :] * dbeta_dkoff

    return np.concatenate((dregressors_dkoff_mm,
                           dregressors_dkon_mm,
                           dregressors_dkoff_others,
                           dregressors_dkon_others,
                           dregressors_dpeq[index_common, :, :]
                          ), axis=0)


def init_kinetic_from_equilibrium(models,
                                  exps,
                                  **opts):
    '''solve eq models, use the result to init kinetic ones'''
    eqi = [i for i, m in enumerate(models) if eq.iseqmodel(m)]
    kii = [i for i, m in enumerate(models) if not eq.iseqmodel(m)]
    if len(eqi) == 0 or len(kii) == 0:
        return
    gammaparam = models[eqi[0]].extra_inputs['gammaparam']
    # convert equilibrium models to kinetic models
    for i in eqi:
        assert models[i].extra_inputs['gammaparam'] == gammaparam, \
            "gammaparam must be consistent across models"
        models[i] = eqmodel2kimodel(models[i],
                                    exp=exps[i],
                                    **opts)
    # create a solver object to organize coefs and params etc.
    seq = mf.regressionsolver([models[i] for i in eqi],
                              optimize_variances=False)
    param_names, param_vals = mf.getsolverparams(seq)

    # apply the stored kons / koffs to the previous kinetic models.
    for i in kii:
        m = models[i]
        assert m.extra_inputs['gammaparam'] == gammaparam, \
            "gammaparam must be consistent across models"
        mf.assign_params(m, param_names, param_vals)
        mf.assign_coefs_fromsolver(m, seq)


def eqmodel2kimodel(eqmodel,
                    tau=0.25,  # 1 / koff in seconds. can be a vector or scalar
                    mintau=1e-4,  # seconds
                    maxtau=20.0,  # seconds
                    minkoff=0.05,
                    maxkoff=1e4,
                    maxkon=1e10,  # sec^-1*M^-1. standard diffusion limit
                    exp=None,
                    **kwargs_unused):
    '''
    convert an equilibrium model to a kinetic one. note that the units of koff
    will always be sec^-1, while the units of kon will depend on the units of
    beta (and ultimately the concentration_unit field of the model)
    '''
    eqmodel = copy.deepcopy(eqmodel)  # in case of later changes to eqmodel
    assert not iskimodel(eqmodel), "already a kinetic model"

    if tau >= maxtau:
        tau = 0.9 * maxtau

    gammaparam = eqmodel.extra_inputs['gammaparam']

    beta_mm, beta_others, basenames_others, index_mm, index_others = \
        eq.getbetas(eqmodel)
    nbindingsteps = beta_mm.size
    nothers = beta_others.size
    assert beta_mm.size > 0 or beta_others.size > 0, "not an eq. model"

    index_common = [i for i in range(eqmodel.nparams)
                    if (i not in index_mm) and (i not in index_others)]

    Ka_others = beta_others
    koff_others = np.ones(nothers) / tau
    for i, basename in enumerate(basenames_others):
        if exp is not None:
            try:
                koff_others[i], _ = \
                    constants.rates_sec_M(basename,
                                          pH=exp['pH'],
                                          T=exp['temperature'])
            except ValueError:
                pass  # stick with koff = 1 / tau
    kon_others = Ka_others * koff_others

    if gammaparam:
        # retrieve gammas directly to prevent roundoff error due to conversion
        # to betas and back again which could cause bound violations
        gamma_mm, gamma_others, _, _, _ = eq.getgammas(eqmodel)
        tau_mm = np.full(nbindingsteps, tau)
        kobs_others = koff_others * (1.0 + gamma_others * Ka_others)
        tau_others = 1.0 / kobs_others

        p = np.concatenate((gamma_mm[0:1], np.diff(gamma_mm),
                            tau_mm, gamma_others, tau_others))
        mingdgd_mm = eqmodel.param_bounds[0][index_mm]
        mingamma_others = eqmodel.param_bounds[0][index_others]
        maxgdg_mm = eqmodel.param_bounds[1][index_mm]
        maxgamma_others = eqmodel.param_bounds[1][index_others]
        lb = np.concatenate((mingdgd_mm,
                             np.full(nbindingsteps, mintau),  # tau_mm
                             mingamma_others,  # gamma_others
                             np.full(nothers, mintau)  # tau_others
                            ))
        ub = np.concatenate((maxgdg_mm,
                             np.full(nbindingsteps, maxtau),  # tau_mm
                             maxgamma_others,  # gamma_others
                             np.full(nothers, maxtau)  # tau_others
                            ))
        gdg_names_mm, tau_names_mm = names.gdgtnames(nbindingsteps)
        gdg_names_others, tau_names_others = names.gdgtnames(basenames_others)
        parameter_names = gdg_names_mm + tau_names_mm + \
            gdg_names_others + tau_names_others
    else:
        Ka_mm = beta_mm / np.append(1.0, beta_mm[:-1])
        koff_mm = np.ones(nbindingsteps) / tau
        kon_mm = Ka_mm * koff_mm
        maxkon_cu = maxkon * eqmodel.concentration_unit
        p = np.concatenate((koff_mm, kon_mm, koff_others, kon_others))
        lb = np.concatenate((np.full(nbindingsteps, minkoff),  # koff_mm
                             np.zeros(nbindingsteps),  # kon_mm
                             np.full(nothers, minkoff),  # koff_others
                             np.zeros(nothers)  # kon_others
                            ))
        ub = np.concatenate((np.full(nbindingsteps, maxkoff),  # koff_mm
                             np.full(nbindingsteps, maxkon_cu),  # kon_mm
                             np.full(nothers, maxkoff),  # koff_others
                             np.full(nothers, maxkon_cu)  # kon_others
                            ))
        koff_names_mm, kon_names_mm = names.rateconstantnames(nbindingsteps)
        koff_names_others, kon_names_others = \
            names.rateconstantnames(basenames_others)
        parameter_names = koff_names_mm + kon_names_mm + \
            koff_names_others + kon_names_others

    # add in extra parameters that are neither kinetic nor eq. constants
    p = np.append(p, eqmodel.p[index_common])
    lb = np.append(lb, eqmodel.param_bounds[0][index_common])
    ub = np.append(ub, eqmodel.param_bounds[1][index_common])
    parameter_names += [eqmodel.parameter_names[i] for i in index_common]

    param_bounds = (lb, ub)

    extra_inputs_ki = dict(regressorfcn_eq=eqmodel.regressorfcn,
                           index_mm=index_mm,
                           index_others=index_others,
                           index_common=index_common,
                           gammaparam=gammaparam,
                           regressorjac_eq=eqmodel.regressorjac,
                           extra_inputs_eq=eqmodel.extra_inputs)

    if eqmodel.regressorjac is None:
        rjac_ki = None
    else:
        rjac_ki = regressorjac_eq2ki

    # note that we've already deepcopied eqmodel
    kimodel = mf.regressionmodel(data=eqmodel.data,
                                 regressorfcn=regressorfcn_eq2ki,
                                 p0=p,
                                 coefs=eqmodel.coefs,
                                 regressorjac=rjac_ki,
                                 param_bounds=param_bounds,
                                 coef_bounds=eqmodel.coef_bounds,
                                 parameter_names=parameter_names,
                                 regressor_names=eqmodel.regressor_names,
                                 data_types=eqmodel.data_types,
                                 scale=eqmodel.scale,
                                 scale_name=eqmodel.scale_name,
                                 auto_scale=eqmodel.auto_scale,
                                 noise_type=eqmodel.noise_type,
                                 modelname=eqmodel.name,
                                 rel_step=eqmodel.rel_step,
                                 extra_inputs=extra_inputs_ki)

    extrafields = ['noise_factor', 'normalization_factor',
                   'concentration_unit', 'time', 'xvals']
    util.copyfields(eqmodel, kimodel, extrafields, dowarn=False)

    return kimodel


def simulate_kinetics(kon, koff,
                      L, c,
                      t,
                      kon_B=1.0, koff_B=1.0,
                      B=0.0, LB=0.0):
    '''
    simulate kinetics of sequential ligand binding to a macromolecule
    in the presence of multiple buffers that each bind one ligand

    Parameters
    ----------
    kon : array
        On rates
    koff : array
        Off rates
    L : float
        Initial ligand concentration
    c : array
        Initial values for [L_jM]
    t : array
        Time points where the state of the system is to be computed
    kon_B : array
        On rates for buffers
    koff_B : array
        Off rates for buffers
    B : array
        Initial values for [B_k]
    LB : array
        Initial values for [LB_k]

    Returns
    -------
    y : 2D array
        concentrations of molecular species, time indexed over columns
    '''

    y0 = np.hstack((c, L, B, LB))
    n = kon.size
    t0 = 0.0

    # initialize arrays which will store rates and their Jacobians
    dy_dt = np.full(y0.size, np.nan)
    J = np.zeros((y0.size, y0.size), dtype=float)

    def f(t, y):
        return sbrate(y, dy_dt, n,
                      kon, koff, kon_B, koff_B)

    def jac(t, y):
        return sbrate_jac(y, J, n,
                          kon, koff, kon_B, koff_B)

    s = ode(f, jac).set_integrator('lsoda', nsteps=2500)  # increase max steps
    s.set_initial_value(y0, t0)

    # system state at each fluorescence measurement
    y = np.zeros((y0.size, t.size), np.float)
    y[:, 0] = y0
    for jj in range(1, t.size):
        # integrate the ODEs forward in time and store the system state at time
        # tobs[jj] in the variable y:
        y[:, jj] = s.integrate(t[jj])

    return y


def stoppedflow_ftimes(m, e):
    return m.model_state['t']


def getrateconstantnames(m):
    return [name for name in m.parameter_names
            if names.israteconstantname(name)]


def getbeta_mm_fromsolver(s):  # fixme probably belongs in another file
    '''
    get beta for the macromolecule, regardless of whether the solver contains
    kinetic or equilbrium models
    '''
    for m in s.models:
        if '\\gamma_{1}' in m.parameter_names:
            u = m.parameter_names.index('\\gamma_{1}')
            gamma = m.p[u:u+1]
            i = 1
            while True:
                i += 1
                name = '\\Delta\\gamma_{{{0}}}'.format(i)
                if name not in m.parameter_names:
                    break
                u = m.parameter_names.index((name))
                gamma = np.append(gamma, gamma[-1] + m.p[u])
            return eq.gamma2beta(gamma)
        elif eq.iseqmodel(m):
            beta_names_mm, _ = eq.get_beta_names([m])
            beta = mf.retrieve_params(m, beta_names_mm)
            return beta
        elif iskimodel(m):
            rcnames = getrateconstantnames(m)
            subnames = [rcn[len('k_{+'):-1] for rcn in rcnames]
            nbindingsteps = np.max([int(name) for name in subnames
                                    if name.isdigit()])
            Ka = np.full(nbindingsteps, np.nan)
            for i in range(1, nbindingsteps + 1):
                j_koff = m.parameter_names.index('k_{{-{0}}}'.format(i))
                j_kon = m.parameter_names.index('k_{{+{0}}}'.format(i))
                koff = m.p[j_koff]
                kon = m.p[j_kon]
                Ka[i - 1] = kon / koff
            beta = np.cumprod(Ka)
            return beta

    raise ValueError("Not a sequential binding model")
