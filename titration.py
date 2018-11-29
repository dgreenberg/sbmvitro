import numpy as np
import equilibrium as eq
import mathfunc
import util
import warnings
import constants
import names


def simulate_mixing(experiment, mixing_model, cu):
    '''calculate nominal ligand and macromolecule concentrations for mixing'''
    assert experiment['titration type'] == 'ligand into macromolecule', \
        "reverse titrations are not yet implemented"
    M0 = experiment['macromolecule_concentration'] / cu
    c_L_other = experiment['total_ligand_other_solution'] / cu

    # rfac is the fraction of the previous volume retained after each injection

    # total injected volumes
    V_M_total = experiment['V_M']
    V_other_total = experiment['V_other']
    # in constrast, V_M and V_other will be volumes remaining in the cell

    injection_size = np.diff(V_M_total + V_other_total)

    if mixing_model == 'cumulative':
        # volume increases, no heat or reagents are lost to ejection
        V_M = V_M_total
        V_other = V_other_total
        V = V_M + V_other
        rfac = np.ones(V.size - 1)
    elif mixing_model == 'integral':
        # volume is constant, injection is instant, no interaction with
        # expelled material
        V_other = V_other_total[0:1]  # dummy injection
        V_M = V_M_total[0:1] - V_other  # remaining macromolecule solution

        V0 = V_M[0] + V_other[0]
        V = np.full(V_M_total.size, V0)
        rfac = np.full(V_M_total.size - 1, np.nan)
        for i in range(injection_size.size):

            # calculate fraction of previous volume retained
            rfac[i] = np.maximum(0.0, V0 - injection_size[i]) / V0

            next_V_M = rfac[i] * V_M[-1] + V_M_total[i + 1] - V_M_total[i]
            next_V_other = rfac[i] * V_other[-1] + \
                V_other_total[i + 1] - V_other_total[i]
            V_M = np.append(V_M, next_V_M)
            V_other = np.append(V_other, next_V_other)
    else:  # fixme add differential, hybrid
        raise ValueError("invalid mixing model")

    Vf_M = V_M / (V_M + V_other)
    Vf_other = V_other / (V_M + V_other)
    Lt = c_L_other * Vf_other
    Mt = M0 * Vf_M

    return Mt, Lt, V, rfac


def titration_params(experiment,
                     nbindingsteps,
                     mingamma=10e-9,  # Molar
                     mindgamma=2.5e-9,  # Molar
                     initial_contamination_macromolecule=0.0,
                     max_contamination_buffer=0.1,
                     max_contamination_mm=np.inf,
                     min_purity_mm=0.5,
                     max_purity_mm=3.0,
                     min_purity_buffer=0.85,
                     max_purity_buffer=1.0,
                     max_affinity_error_buffer=0.5,  # vs. constants.py
                     maxgamma=np.inf,  # max mm gamma for initialization only
                     maxdgamma=np.inf,
                     concentration_unit=1e-6,
                     gammaparam=True,
                     buffer_contamination=False,
                     macromolecule_contamination=True,
                     buffer_purity=True,
                     macromolecule_purity=True,
                     unique_mm_contamination=None,
                     mixing_model='integral',
                     **kwargs_unused):
    '''
    get initial values, bounds and names for standard titration parameters.
    also returns extra_inputs dict for regressorfcn and regressorjac
    '''
    fromLf = 'Lf' in experiment.keys()
    hasbuffer = 'buffer' in experiment.keys() and \
        experiment['buffer'] != 'none'
    mm, ll = experiment['macromolecule'], experiment['ligand']
    cu = concentration_unit
    if unique_mm_contamination is None:
        unique_mm_contamination = not hasbuffer

    extra_inputs \
        = dict(nbindingsteps=nbindingsteps,
               gammaparam=gammaparam,
               buffer_purity=buffer_purity,
               buffer_contamination=buffer_contamination,
               macromolecule_purity=macromolecule_purity,
               macromolecule_contamination=macromolecule_contamination)

    if fromLf:
        extra_inputs['Lf'] = experiment['Lf'] / cu
        gamma_mm = eq.guess_gamma_fromLf(nbindingsteps,
                                         extra_inputs['Lf'], maxgamma=maxgamma,
                                         maxdgamma=maxdgamma / cu)
    else:
        if experiment['type'] == 'spectra':
            ndata = experiment['data'].shape[1]
            Lt = util.asvec(experiment['total ligand'], ndata) / cu
            Mt = util.asvec(experiment['total macromolecule'], ndata) / cu
        elif experiment['type'] == 'itc':
            Mt, Lt, V, rfac = simulate_mixing(experiment, mixing_model, cu)
            extra_inputs['V'] = V
            extra_inputs['rfac'] = rfac
            ndata = V.size
        else:
            raise ValueError("experiment is not a titration")
        extra_inputs.update({'Lt_nominal': Lt, 'Mt_nominal': Mt})

        if hasbuffer:
            Bt = util.asvec(experiment['total buffer'], ndata) / cu
            extra_inputs['Bt_nominal'] = Bt
            bb = experiment['buffer']
            try:
                IS_mm = experiment['ionic strength'] * 1e3
                kd_B_nm = constants.kd_nM(bb,
                                          pH=experiment['pH'],
                                          T=experiment['temperature'],
                                          IS_mm=IS_mm)
                gamma_B = kd_B_nm * 1e-9 / cu
                # guess Lf from the buffer and total ligand, ignoring mm
                Lf_Bonly = eq.Lt2Lf_singlebuffer(np.atleast_1d(1.0 / gamma_B),
                                                 Lt, Bt)[0]
                gamma_mm = eq.guess_gamma_fromLf(nbindingsteps, Lf_Bonly,
                                                 maxgamma=maxgamma / cu,
                                                 maxdgamma=maxdgamma / cu)
                min_gamma_B = gamma_B * (1.0 - max_affinity_error_buffer)
                max_gamma_B = gamma_B * (1.0 + max_affinity_error_buffer)
                assert max_affinity_error_buffer < 1.0, "invalid setting"

            except ValueError:
                warnings.warn("failed to retrieve rate constants for {0}"
                              .format(bb))
                gamma_mm = eq.guess_gamma(nbindingsteps,
                                          Mt,
                                          Lt,
                                          maxgamma=maxgamma / cu,
                                          mingamma=mingamma / cu,
                                          maxdgamma=maxdgamma / cu)
                gamma_B = np.mean(gamma_mm)  # wild guess
                min_gamma_B = mindgamma / cu
                max_gamma_B = np.inf
        else:
            gamma_mm = eq.guess_gamma(nbindingsteps,
                                      Mt,
                                      Lt,
                                      maxgamma=maxgamma / cu,
                                      mingamma=mingamma / cu,
                                      maxdgamma=maxdgamma / cu)
    if gammaparam:
        p0 = np.append(gamma_mm[0], np.diff(gamma_mm))
        parameter_names, _ = names.gdgtnames(nbindingsteps)
        lb = np.append(mingamma / cu,
                       np.full(nbindingsteps - 1, mindgamma / cu))
        ub = np.append(maxgamma / cu,
                       np.full(nbindingsteps - 1, maxdgamma / cu))
    else:
        p0 = eq.gamma2beta(gamma_mm)
        parameter_names = names.betanames(nbindingsteps)
        lb = np.zeros(nbindingsteps)
        ub = np.full(nbindingsteps, np.inf)

    if fromLf:
        return p0, parameter_names, (lb, ub), extra_inputs

    # mm purity
    if macromolecule_purity:
        p0 = np.append(p0, 1.0)
        lb = np.append(lb, min_purity_mm)
        ub = np.append(ub, max_purity_mm)
        if 'protein_purification' in experiment.keys():
            pp = experiment['protein_purification']
            parameter_names.append('{0} purity ({1})'.format(mm, pp))
        else:
            parameter_names.append('{0} purity'.format(mm))
    # mm contamination
    if macromolecule_contamination:
        p0 = np.append(p0, initial_contamination_macromolecule)
        pname = '{0} in {1} contamination ratio'.format(ll, mm)
        if 'protein_purification' in experiment.keys():
            pname += ' ({0})'.format(pp)
        if unique_mm_contamination:
            pname += '@'  # unique value for each model
        parameter_names.append(pname)
        lb = np.append(lb, 0.0)
        ub = np.append(ub, max_contamination_mm)

    if not hasbuffer:
        return p0, parameter_names, (lb, ub), extra_inputs

    # beta or gamma
    if gammaparam:
        p0 = np.append(p0, gamma_B)  # gamma = K_d
        lb = np.append(lb, min_gamma_B)
        ub = np.append(ub, max_gamma_B)
        parameter_names.append('\\gamma_{{{0}}}'.format(bb))
    else:
        p0 = np.append(p0, 1.0 / gamma_B)  # gamma = K_d
        lb = np.append(lb, 0.0)
        ub = np.append(ub, np.inf)
        parameter_names.append('\\beta_{{{0}}}'.format(bb))

    # buffer purity
    if buffer_purity:
        p0 = np.append(p0, 1.0)
        lb = np.append(lb, min_purity_buffer)
        ub = np.append(ub, max_purity_buffer)
        if 'buffer_batch' in experiment.keys():
            parameter_names.append('{0} purity (batch {1})'
                                   .format(bb, experiment['buffer_batch']))
        else:
            parameter_names.append('{0} purity (unknown batch)@'.format(bb))
    # buffer contamination
    if buffer_contamination:
        p0 = np.append(p0, 0.0)
        lb = np.append(lb, 0.0)
        ub = np.append(ub, max_contamination_buffer)
        if 'buffer_batch' in experiment.keys():
            parameter_names.append('{0} in {1} contamination ratio (batch {2})'
                                   .format(ll, bb, experiment['buffer_batch']))
        else:
            parameter_names.append('{0} in {1} contamination '
                                   'ratio (unknown batch)@'.format(ll, bb))

    return p0, parameter_names, (lb, ub), extra_inputs


def decompose_p_titration(p,
                          nbindingsteps,
                          gammaparam,
                          fromLf,
                          buffered,
                          buffer_purity,
                          buffer_contamination,
                          macromolecule_purity,
                          macromolecule_contamination):
    if gammaparam:
        gamma = p[:nbindingsteps].cumsum()
        beta = eq.gamma2beta(gamma)
    else:
        beta = p[:nbindingsteps]
    offset = nbindingsteps

    if fromLf:  # free ligand concentration known in advance
        purity_M, contamination_M, beta_B, purity_B, contamination_B = \
            None, None, None, None, None
    else:
        if macromolecule_purity:
            purity_M = p[offset]
            offset += 1
        else:
            purity_M = 1.0
        if macromolecule_contamination:
            contamination_M = p[offset]
            offset += 1
        else:
            contamination_M = 0.0
        if buffered:
            if gammaparam:
                beta_B = 1.0 / p[offset]  # gamma = K_d for monovalent buffers
            else:
                beta_B = p[offset]
            offset += 1
            if buffer_purity:
                purity_B = p[offset]
                offset += 1
            else:
                purity_B = 1.0
            if buffer_contamination:
                contamination_B = p[offset]
                offset += 1
            else:
                contamination_B = 0.0
        else:
            beta_B, purity_B, contamination_B = None, None, None

    return beta, purity_M, contamination_M, beta_B, purity_B, contamination_B


def regressorfcn_titration(p,
                           nbindingsteps=None,
                           gammaparam=None,
                           Lf=None,
                           Mt_nominal=None,
                           Lt_nominal=None,
                           Bt_nominal=None,
                           buffer_purity=None,
                           buffer_contamination=None,
                           macromolecule_purity=None,
                           macromolecule_contamination=None):
    fromLf, buffered = Lf is not None, Bt_nominal is not None
    beta, purity_M, contamination_M, beta_B, purity_B, contamination_B = \
        decompose_p_titration(p, nbindingsteps, gammaparam, fromLf, buffered,
                              buffer_purity,
                              buffer_contamination,
                              macromolecule_purity,
                              macromolecule_contamination)

    if fromLf:  # free ligand concentration known in advance

        F, Lfpowers, _, R = eq.Lf2bsfracs(beta, Lf)
        state = {'bsfracs': F, 'R': R, 'Lfpowers': Lfpowers}
        return F, state

    else:  # need to infer free ligand from totals

        Mt = Mt_nominal * purity_M
        Lt = Lt_nominal + Mt_nominal * contamination_M

        if buffered:

            Bt = Bt_nominal * purity_B
            Lt += Bt_nominal * contamination_B

            beta_all = [beta, np.atleast_1d(beta_B)]
            Mt_all = np.append(Mt.reshape(-1, 1), Bt.reshape(-1, 1), axis=1)
            bsfracs, P, Lfpowers, R, Rterms = \
                eq.binding_equilibrium(beta_all, Lt, Mt_all)

            # binding state concentrations:
            LjM = bsfracs[0, :, :] * Mt.reshape(-1, 1)

        else:  # not buffered

            Lf, P = eq.Lt2Lf_singlebuffer(beta, Lt, Mt)
            bsfracs, Lfpowers, Rterms, R = eq.Lf2bsfracs(beta, Lf)
            LjM = bsfracs * Mt.reshape(-1, 1)
            Bt = None

        state = dict(LjM=LjM,
                     P=P,
                     R=R,
                     Rterms=Rterms,
                     bsfracs=bsfracs,
                     Mt=Mt,
                     Lt=Lt,
                     Lfpowers=Lfpowers,
                     Bt=Bt)

        return LjM, state


def regressorjac_titration(p, state,
                           nbindingsteps=None,
                           gammaparam=None,
                           Lf=None,
                           Mt_nominal=None,
                           Lt_nominal=None,
                           Bt_nominal=None,
                           buffer_purity=None,
                           buffer_contamination=None,
                           macromolecule_purity=None,
                           macromolecule_contamination=None):
    fromLf, buffered = Lf is not None, Bt_nominal is not None
    nr = nbindingsteps + 1  # nregressors

    beta, purity_M, contamination_M, beta_B, purity_B, contamination_B = \
        decompose_p_titration(p, nbindingsteps, gammaparam, fromLf, buffered,
                              buffer_purity,
                              buffer_contamination,
                              macromolecule_purity,
                              macromolecule_contamination)

    if gammaparam:
        gamma = p[:nbindingsteps].cumsum()
        # derivative of beta w.r.t. gamma:
        db_dg = eq.dbeta_dgamma(gamma)
        # derivative of beta w.r.t np.append(gamma[0], np.diff(gamma)):
        db_dg0dg = np.dot(db_dg, np.tri(nbindingsteps))

    Lfpowers, R = state['Lfpowers'], state['R']

    if fromLf:

        J = eq.dbsfracs_dbeta(beta,
                              Lfpowers,
                              R,
                              state['bsfracs'])

    else:

        m = Lt_nominal.size
        F, P, Lt, Mt = state['bsfracs'], state['P'], state['Lt'], state['Mt']

        if buffered:

            Bt, Rterms = state['Bt'], state['Rterms']
            bsfracs_mm = state['bsfracs'][0, :, :]
            beta_all = [beta, np.atleast_1d(beta_B)]
            Mt_all = np.append(Mt.reshape(-1, 1), Bt.reshape(-1, 1), axis=1)
            ii = np.arange(nr).reshape(-1, 1)
            saturation = np.concatenate((np.dot(F[0, :, :], ii),
                                         np.dot(F[1, :, :], ii)),
                                        axis=1)

            # calculate derivatives of free ligand w.r.t various quantities
            dLf_dLt, dLf_dMt_all, dLf_dbeta = \
                eq.Lfgradients_multibuffer(beta_all, Lt, Mt_all,
                                           Lfpowers, Rterms, R, saturation)

            # partial derivatives of macromolecule binding state fractions
            # w.r.t free ligand concentration
            dF_dLf = eq.dbsfracs_dLf(beta, Lfpowers, R[:, 0])

            # partial derivatives of macromolecule binding state fracs
            # w.r.t. macromolecule beta
            dF_dbeta_partial = \
                eq.dbsfracs_dbeta(beta, Lfpowers, R[:, 0], bsfracs_mm)

            # derivatives w.r.t. total ligand
            dF_dLt = dF_dLf * dLf_dLt.reshape(-1, 1)
            dbscons_dLt = dF_dLt * Mt.reshape(-1, 1)

            # derivatives of mm binding state fractions w.r.t. beta_mm
            dF_dbeta_mm = np.empty((nbindingsteps, m, nr))
            for i in range(nbindingsteps):
                dF_dbeta_mm[i, :, :] = dF_dbeta_partial[i, :, :] + \
                    dF_dLf * dLf_dbeta[0][:, i:i + 1]
            dbscons_dbeta_mm = dF_dbeta_mm * Mt.reshape(-1, 1)

            if macromolecule_purity:
                # derivatives of mm binding state fractions
                # w.r.t total macromolecule
                dF_dMt = dF_dLf * dLf_dMt_all[:, 0:1]
                # bscons = bsfracs * Mt; use the product rule
                dbscons_dMt = dF_dMt * Mt.reshape(-1, 1) + bsfracs_mm
                # Mt = Mt_nominal * purity_M; use the chain rule
                dbscons_dpurity_M = dbscons_dMt * Mt_nominal.reshape(-1, 1)
            else:
                dbscons_dpurity_M = np.zeros(0)
            # Lt = Mt_nominal * contamination_M + ... ; use the chain rule
            if macromolecule_contamination:
                dbscons_dcontamination_M = dbscons_dLt * \
                    Mt_nominal.reshape(-1, 1)
            else:
                dbscons_dcontamination_M = np.zeros(0)

            # same logic for the buffer:
            dF_dbeta_B = dF_dLf * dLf_dbeta[1][:, 0:1]
            dbscons_dbeta_B = dF_dbeta_B * Mt.reshape(-1, 1)
            if buffer_purity:
                dF_dBt = dF_dLf * dLf_dMt_all[:, 1:2]
                dbscons_dBt = dF_dBt * Mt.reshape(-1, 1)
                dbscons_dpurity_B = dbscons_dBt * Bt_nominal.reshape(-1, 1)
            else:
                dbscons_dpurity_B = np.zeros(0)
            if buffer_contamination:
                dbscons_dcontamination_B = \
                    dbscons_dLt * Bt_nominal.reshape(-1, 1)
            else:
                dbscons_dcontamination_B = np.zeros(0)

            J = np.concatenate((dbscons_dbeta_mm,
                                dbscons_dpurity_M.reshape(-1, m, nr),
                                dbscons_dcontamination_M.reshape(-1, m, nr),
                                dbscons_dbeta_B.reshape(-1, m, nr),
                                dbscons_dpurity_B.reshape(-1, m, nr),
                                dbscons_dcontamination_B.reshape(-1, m, nr)
                                ), axis=0)

        else:

            # get partial derivatives of macromolecule binding state fractions
            # w.r.t free ligand concentration
            dF_dLf = eq.dbsfracs_dLf(beta, Lfpowers, R)
            dLf_dP = mathfunc.droot_dpolynomial(P, Lfpowers[:, 1])

            dbscons_dbeta = eq.dF_dbeta_singlebuffer(beta,
                                                     dLf_dP,
                                                     dF_dLf,
                                                     Lfpowers,
                                                     R,
                                                     F,
                                                     Lt,
                                                     Mt) * Mt.reshape(-1, 1)

            if macromolecule_purity:
                # derivatives w.r.t. base macromolecule concentration. We use
                # the chain rule and the fact that Mt = purity_M * Mt_nominal
                dF_dpurity_M = \
                    eq.dF_dMt_singlebuffer(beta,
                                           dLf_dP,
                                           dF_dLf) * Mt_nominal.reshape(-1, 1)
                # bscons = F * Mt = F * purity_M * Mt_nominal; use  product rule
                dbscons_dpurity_M = \
                    (dF_dpurity_M * purity_M + F) * Mt_nominal.reshape(-1, 1)
            else:
                dbscons_dpurity_M = np.zeros(0)

            if macromolecule_contamination:
                '''derivatives w.r.t. contamination
                we note that Lt = Lt_nominal + contamination_M * Mt_nominal
                and use the chain rule'''
                dF_dcontamination_M = \
                    eq.dF_dLt_singlebuffer(beta,
                                           dLf_dP,
                                           dF_dLf) * Mt_nominal.reshape(-1, 1)
                dbscons_dcontamination_M = \
                    dF_dcontamination_M * Mt_nominal.reshape(-1, 1)
            else:
                dbscons_dcontamination_M = np.zeros(0)  # we'll reshape it

            J = np.concatenate((dbscons_dbeta,
                                dbscons_dpurity_M.reshape(-1, m, nr),
                                dbscons_dcontamination_M.reshape(-1, m, nr)),
                               axis=0)

    if gammaparam:  # change of variables from beta -> gamma0 / dgamma
        J[:nbindingsteps, :, :] = \
            np.dot(db_dg0dg.T,
                   J[:nbindingsteps, :, :].reshape(nbindingsteps, -1)
                   ).reshape(nbindingsteps, -1, nr)
        if buffered:
            dbeta_B_dgamma_B = -(beta_B ** 2)
            offset = nbindingsteps + 1 + int(macromolecule_contamination)
            J[offset, :, :] *= dbeta_B_dgamma_B

    return J
