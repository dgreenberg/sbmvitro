import copy
import titration
import modelfit as mf
import names


def itc2sbmodel(experiment,
                nbindingsteps,
                **opts):
    opts.setdefault('concentration_unit', 1e-6)
    opts.setdefault('mixing_model', 'integral')
    experiment = copy.deepcopy(experiment)

    # general titration stuff:
    p0, parameter_names, param_bounds, extra_inputs = \
        titration.titration_params(experiment, nbindingsteps, **opts)

    # itc-specific stuff:
    data_types = ['heat']
    regressor_names = names.itcregressornames(experiment, nbindingsteps)

    m = mf.regressionmodel(experiment['data'].reshape(-1, 1),
                           regressorfcn_itc,
                           p0,
                           regressorjac=regressorjac_itc,
                           parameter_names=parameter_names,
                           regressor_names=regressor_names,
                           data_types=data_types,
                           scale_name='itc scale',
                           noise_type='itc noise',  # T-dependent? fixme
                           param_bounds=param_bounds,
                           coef_bounds=None,  # enthalpies are unbounded
                           extra_inputs=extra_inputs)

    m.concentration_unit = opts['concentration_unit']

    return m


def regressorfcn_itc(p, V=None, rfac=None, **extra_inputs):
    '''
    regressors are concentration differences of mm states with 1+ ligand bound
    '''
    LjM, state = titration.regressorfcn_titration(p, **extra_inputs)
    bsmoles = LjM * V.reshape(-1, 1)
    # FIXME: the regressors are correct for forward
    # but not for reverse titrations
    regressors = bsmoles[1:, 1:] - bsmoles[:-1, 1:] * rfac.reshape(-1, 1)
    return regressors, state


def regressorjac_itc(p, state, V=None,
                     rfac=None,
                     **extra_inputs):
    dLjM_dp = titration.regressorjac_titration(p, state, **extra_inputs)
    dbsmoles_dp = dLjM_dp * V.reshape(1, -1, 1)
    dregressors_dp = dbsmoles_dp[:, 1:, 1:] - \
        dbsmoles_dp[:, :-1, 1:] * rfac.reshape(1, -1, 1)
    return dregressors_dp
