import os
import pickle
import modelfit as mf
import numpy as np
import equilibrium as eq
import kinetics as ki
import itc
import spectra
import import_all_data as iad
import ivdatatools as ivd
from sensitivity import sbfit_sensitivity
from multistart import sbfit_multistart
import graphs as gr


default_residual_scales = dict(itc=0.43950682914233818,
                               spectra=3203.8721009569322,
                               stoppedflow=0.062303237143837835)


def gf_defaultopts():
    return dict(concentration_unit=1e-6,
                maxgamma=1e-6,
                mint_stoppedflow=1e-3,
                maxdgamma=1e-6,
                min_wavelength=0.0,
                max_wavelength=np.inf,
                maxtau=1.5,
                mintau=1e-4,
                gammaparam=True,
                macromolecule_contamination=True,
                max_contamination_buffer=0.1,
                buffer_contamination=True,
                max_affinity_error_buffer=0.05,
                min_purity_buffer=0.75,
                max_purity_buffer=1.0,
                min_purity_mm=0.9,
                max_purity_mm=1.1)


def sbglobalsolver(experiments, nbindingsteps,
                   fluorescence_groups=None,
                   reweight=True,
                   **opts_in):
    assert len(experiments) > 0, "no data"
    opts = gf_defaultopts()
    opts.update(opts_in)

    models = experiments2models(experiments, nbindingsteps, **opts)

    # reweight the data to account different units and numbers of observations
    # so that each experiment type has roughly the same influence on the fit:
    if reweight:
        models = mf.reweight_data(models, experiments,
                                  residual_scales=default_residual_scales)

    if fluorescence_groups is not None:
        for i, m in enumerate(models):
            if experiments[i]['type'] == 'itc':
                continue
            models[i] = mf.collapse_regressors(m, fluorescence_groups)

    eqi = [i for i, m in enumerate(models) if eq.iseqmodel(m)]
    if np.any(eqi) and (not np.all(eqi)):
        ki.init_kinetic_from_equilibrium(models, experiments, **opts)

    s = mf.regressionsolver(models, optimize_variances=False)
    s.optimize_scales_allmodels()
    # note we don't optimize coefs yet
    return s, opts


def experiments2models(experiments, nbindingsteps, **opts):
    '''
    convert all experiments to SBM models of the appropriate types
    '''
    models = []
    for exp in experiments:
        etype = exp['type']
        if etype == 'spectra':
            model = spectra.spectra2sbmodel(exp, nbindingsteps, **opts)
        elif etype == 'itc':
            model = itc.itc2sbmodel(exp, nbindingsteps, **opts)
        elif etype == 'stoppedflow':
            model = ki.stoppedflow2sbmodel(exp, nbindingsteps, **opts)
        else:
            raise ValueError("unknown experiment type: {0}".format(etype))
        models.append(model)

    return models


def preparesolver(nbindingsteps=4, temp=37.0, fluorescence_groups=None,
                  include_nuisance_params=True, reweight=True, **opts_in):

    opts = dict(initial_contamination_macromolecule=1.0,
                maxtau=5.0,
                maxF_stoppedflow=6.5,  # volts, to prevent PMT saturation
                max_spec_f=1e6,  # photos/s, max 4e6, to avoid saturation
                max_contamination_mm=10.0,  #ratop of total Ca2+ to nominal mm
                maxgamma=500e-9,
                maxdgamma=1.5e-6,
                min_wavelength=350,  # below 350 BAPTA absorbs (ca-dependent)
                fluorescence_groups=fluorescence_groups)
    opts.update(opts_in)

    datadir = ivd.autodatadir()
    experiments = iad.import_all(datadir)
    requirements = {'temperature': temp,
                    'type': ['spectra', 'stoppedflow', 'itc'],
                    'quality': [1, 2],
                    'buffer': 'bapta',  # when key is present
                    'macromolecule': 'GCaMP6s',
                    'ligand': 'Ca2+',
                    'titration type': 'ligand into macromolecule',
                    'reference corrected': True,
                    'protein_purification': ['pu5', 'pu6', 'pu7', 'pu12']}
    elist = ivd.filter_experiments(experiments, **requirements)
    elist = [e for e in elist if e['type'] != 'spectra' or
             e['data'].max() <= opts['max_spec_f']]
    elist = [e for e in elist if e['type'] != 'stoppedflow' or
             e['protein_purification'] != 'pp_1708']

    if temp == 22.0:
        target_spectra = ['spectra 160223 GCaMP bapta batch2 22C.csv',
                          'GCaMP6s_fluorescence spectra set2.csv']
        egood = [e for e in elist
                 if e['type'] in ['itc', 'stoppedflow'] or
                 os.path.basename(e['filename']) in target_spectra]
    elif temp == 37.0:
        '''
        subd = os.path.join(datadir, 'Stopped-Flow',
                            'on rates - BAPTA buffer system', '37C',
                            'set 1 - 160303', 'CSV')
        target_sf = \
            [os.path.join(datadir, 'Stopped-Flow', 'off rates', '37C', 'CSV',
                          '3_all traces 160308.csv'),
             os.path.join(subd, 'all traces sample 1.csv'),  # no calcium
             os.path.join(subd, 'all traces sample 2.csv'),
             os.path.join(subd, 'all traces sample 3.csv'),
             os.path.join(subd, 'all traces sample 4.csv'),
             os.path.join(subd, 'all traces sample 5.csv'),
             os.path.join(subd, 'all traces sample 6.csv')]

        egood = [e for e in elist
                 if e['type'] != 'stoppedflow' or e['filename'] in target_sf]
        '''
        egood = elist
    else:
        egood = elist

    fixed_params = []
    if not include_nuisance_params:
        opts['buffer_contamination'] = False
        opts['macromolecule_contamination'] = False
        opts['buffer_purity'] = False
        opts['macromolecule_purity'] = False
        # Fix buffer K_d's.
        # We don't fix time constants since the literature is so weak there.
        # fixme: only do this for buffers in constants.py
        buffers = list(set([e['buffer'] for e in egood
                            if 'buffer' in e.keys()]))
        fixed_params = ['\\gamma_{{{0}}}'.format(b) for b in buffers]

    s, optsout = sbglobalsolver(egood, nbindingsteps, reweight=reweight, **opts)
    optsout['reweight'] = reweight

    return s, fixed_params, egood, optsout


def get_nuisance_params(solver, experiments):
    params = [n for i, n in enumerate(solver.x_names) if solver.isparam[i]]
    nuisance_params = \
        [n for n in params if 'purity' in n or 'contamination' in n]
    buffers = list(set([e['buffer'] for e in experiments
                        if 'buffer' in e.keys()]))
    for buf in buffers:
        nuisance_params += [n for n in params if buf in n]
    extras = ['dead time']
    for ext in extras:
        nuisance_params += [n for n in params if ext in n]
    return list(np.unique(nuisance_params))


if __name__ == '__main__':
    # only run this code if we're not importing this as a module
    nbindingsteps = 4
    s, fixed_params, egood, opts = preparesolver(nbindingsteps=nbindingsteps)

    multistart_init_ranges = {'\\tau_{1}':(None, 1),'\\tau_{2}':(None, 1),
                              '\\tau_{3}':(None, 1), '\\tau_{4}':(None, 1)}

    #task = 'multistart_reload'
    #task = 'none'
    task = 'multistart_solve'
    #'multistart_solve_invivosubmodel'
    print("Executing task: {0}".format(task))

    if task == 'solve_from_invivo_rcs':

        xvals = s.x.copy()
        koff = np.array([3.7289, 13.5937, 85.1815, 0.92908])
        kon = np.array([10.998, 0.61723, 1753.6, 9.9354])
        # fixme shouldn't assume param order
        if s.models[0].extra_inputs['gammaparam']:
            gamma, tau = ki.koffkon2gammatau_bs(koff, kon)
            xvals[:nbindingsteps] = np.append(gamma[0], np.diff(gamma))
            xvals[nbindingsteps:2 * nbindingsteps] = tau
        else:
            xvals[:nbindingsteps] = koff
            xvals[nbindingsteps:2 * nbindingsteps] = kon
        xorig = xvals.copy()
        xvals = np.minimum(np.maximum(xvals, s.bounds[0]), s.bounds[1])
        r = xorig / xvals
        r[np.isnan(r)] = 1.0
        minr = np.minimum(1.0, np.min(r))
        maxr = np.maximum(1.0, np.max(r))
        dxx = np.maximum(maxr - 1.0, 1.0 - minr)
        if dxx != 0:
            print("max parameter adjustment: {0}".format(dxx))

        s.loglikelihood(xvals)  # assign in vivo rate constants
        xpmask = np.ones(s.isparam.sum(), dtype=bool)
        xpmask[:2 * nbindingsteps] = False

        s.solve_profile(xpmask=xpmask)

    elif task == 'multistart_solve_invivosubmodel':
        assert np.all([m.extra_inputs['gammaparam'] for m in s.models]), \
            "gammaparam must be True for all models"
        for m in s.models:
            i = m.parameter_names.index('\\Delta\\gamma_{2}')
            m.param_bounds[1][i] = np.minimum(m.param_bounds[1][i], 0.1)
            m.p[i] = np.minimum(m.param_bounds[1][i], m.p[i])
            i = m.parameter_names.index('\\Delta\\gamma_{4}')
            m.param_bounds[1][i] = np.minimum(m.param_bounds[1][i], 0.01)
            m.p[i] = np.minimum(m.param_bounds[1][i], m.p[i])
            if '\\tau_{2}' not in m.parameter_names:
                continue
            i = m.parameter_names.index('\\tau_{2}')
            m.param_bounds[1][i] = np.minimum(m.param_bounds[1][i], 0.005)
            m.p[i] = np.minimum(m.param_bounds[1][i], m.p[i])
            i = m.parameter_names.index('\\tau_{4}')
            m.param_bounds[1][i] = np.minimum(m.param_bounds[1][i], 0.005)
            m.p[i] = np.minimum(m.param_bounds[1][i], m.p[i])

        s = mf.regressionsolver(s.models, deepcopy=True,
                                optimize_variances=False)
        s.loglikelihood(s.x)

        multistart_init_ranges = {'\\Delta\\gamma_{3}': (None, 0.6),
                                  '\\tau_{1}': (None, 0.8),
                                  '\\tau_{3}': (None, 0.8)}

        x_optimized, LL_optimized, niter_used, all_param_vals, all_LL = \
            sbfit_multistart(s, nbindingsteps,
                             multistart_init_ranges=multistart_init_ranges,
                             progressfile='fit50par.pickle',
                             save_progress=True, fixed_params=fixed_params)

    elif task == 'multistart_solve':
        nmultistarts = 50
        nmultithreads = 15
        x_optimized, LL_optimized, niter_used, all_param_vals, all_LL = \
            sbfit_multistart(s, nbindingsteps, nstarts=nmultistarts,
                             nworkers=nmultithreads,
                             multistart_init_ranges=multistart_init_ranges,
                             save_progress=True, fixed_params=fixed_params)
        with open('fit50par.pickle', 'wb') as fid:
            pickle.dump(x_optimized, fid)
            pickle.dump(LL_optimized, fid)
            pickle.dump(niter_used, fid)
            pickle.dump(all_param_vals, fid)
            pickle.dump(all_LL, fid)
            pickle.dump(s.x_names, fid)
            pickle.dump(opts, fid)
            pickle.dump(egood, fid)
        gr.plot_globalfit(s, egood)

    elif task == 'multistart_reload':

        fn_reload = 'fit50par_040118_SWdata.pickle'
        #fn_reload = 'fit50par new sf data only all other data.pickle'
        #fn_reload = 'fit50par.pickle'

        with open(fn_reload, 'rb') as fid:
            x_optimized = pickle.load(fid)
            LL_optimized = pickle.load(fid)
            niter_used = pickle.load(fid)
            all_param_vals = pickle.load(fid)
            all_LL = pickle.load(fid)
            x_names = pickle.load(fid)
            opts = pickle.load(fid)
            egood = pickle.load(fid)
            bestind = LL_optimized.argmax()
            s.loglikelihood(x_optimized[bestind, :])


    elif task == 'sensitivity_analysis':

        LL_perturbed, perturbed_param_vals, perturbed_x = \
            sbfit_sensitivity(s,
                              nbindingsteps,
                              max_changefrac=0.1,
                              nsteps=5)

    elif task == 'sensitivity_analysis_nuisancefixed':
        LL_perturbed, perturbed_param_vals, perturbed_x = \
            sbfit_sensitivity(s,
                              nbindingsteps,
                              max_changefrac=0.1,
                              nsteps=5,
                              fixed_params=get_nuisance_params(s, egood))
    elif task == 'none':
        print("no task")
    else:
        raise ValueError("Invalid task: {0}".format(task))
