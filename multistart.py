import os
from multiprocessing import Pool, cpu_count
import time
import pickle
import numpy as np
import names
import pyDOE


def sbfit_multistart(s, nbindingsteps, fixed_params=(), **opts):
    xpnames = list(np.array(s.x_names)[s.isparam])
    msn = [pname for pname in names.binding_param_names(nbindingsteps)
           if pname not in fixed_params and pname in xpnames]
    return solve_profile_multistart(s, msn, fixed_params=fixed_params, **opts)


def solve_profile_multistart(s, multistart_params,
                             multistart_init_ranges=None,
                             nstarts=50,
                             fixed_params=(),
                             multiprocess=True,
                             nworkers=None,
                             save_progress=False,
                             progressfile='multistart progress.pickle',
                             verbose=100,
                             maxiter_init=10,
                             maxiter_opt=5000,
                             print_every=10):
    '''
    do a a grid of initializations for the chosen binding parameters and
    return the solver with the best solution chosen
    '''
    multistart_xi = np.array([], dtype=int)
    for pname in multistart_params:
        if pname in s.x_names and s.isparam[s.x_names.index(pname)]:
            multistart_xi = np.append(multistart_xi, s.x_names.index(pname))
    n_multistart_params = len(multistart_xi)
    assert(n_multistart_params > 0), "no valid params for multistart"
    assert not (np.isinf(s.bounds[0][multistart_xi]).any() or
                np.isinf(s.bounds[1][multistart_xi]).any()), \
        "bounds must be set on all multistart params"

    lb = s.bounds[0][multistart_xi]
    ub = s.bounds[1][multistart_xi]
    if multistart_init_ranges is not None:
        assert isinstance(multistart_init_ranges, dict), "invalid input"
        for pname in multistart_init_ranges.keys():
            if pname not in multistart_params:
                continue
            i = multistart_params.index(pname)
            if multistart_init_ranges[pname][0] is not None \
                and not np.isnan(multistart_init_ranges[pname][0]):

                lb[i] = multistart_init_ranges[pname][0]

            if multistart_init_ranges[pname][1] is not None \
                and not np.isnan(multistart_init_ranges[pname][1]):

                ub[i] = multistart_init_ranges[pname][1]
    assert np.all(lb >= s.bounds[0][multistart_xi]) and \
        np.all(ub <= s.bounds[1][multistart_xi]), "init ranges violate bounds"
    assert not (np.isinf(lb).any() or np.isinf(ub).any()), "finite ranges required"

    # Latin hypercube sampling
    zvals = pyDOE.lhs(n_multistart_params, nstarts, criterion='c')

    LL_optimized = np.full(nstarts, np.nan)
    x_optimized = np.full((nstarts, s.nx), np.nan)
    niter_used = np.zeros(nstarts, dtype=int)
    all_param_vals, all_LL_vals = [], []

    if multiprocess:
        pfbase, pfext = os.path.splitext(progressfile)
        sbf_inputs = []
        progressfiles_multiprocess = []
        for i in range(nstarts):
            xvals = lb + (ub - lb) * zvals[i, :]
            progressfile_thisi = os.path.join(os.getcwd(),
                                              pfbase + ' {0}'.format(i) + pfext)
            progressfiles_multiprocess.append(progressfile_thisi)
            sbf_inputs.append(dict(s0=s, xi=multistart_xi, xvals=xvals,
                                   maxiter_init=maxiter_init,
                                   maxiter_opt=maxiter_opt,
                                   print_every=print_every,
                                   fixed_params=fixed_params,
                                   save_progress=save_progress,
                                   progressfile=progressfile_thisi,
                                   verbose=0, index=i))

        if nworkers is None:
            nworkers = np.minimum(cpu_count(), nstarts)

        t0 = time.time()
        with Pool(nworkers) as p:
            results = p.map(sbfit_ps_wrapper, sbf_inputs)
        t1 = time.time()
        if verbose > 3:
            print("Elapsed (wall clock) time: {0} seconds".format(t1 - t0))

        for i, r in enumerate(results):
            x_optimized[i, :] = r['x']
            LL_optimized[i] = r['LL']
            niter_used[i] = r['niter']
            all_param_vals.append(r['xp_all'])
            all_LL_vals.append(r['LL_all'])

        if save_progress:
            with open(progressfile, 'wb') as fid:
                pickle.dump(x_optimized, fid)
                pickle.dump(LL_optimized, fid)
                pickle.dump(niter_used, fid)
                pickle.dump(all_param_vals, fid)
                pickle.dump(all_LL_vals, fid)
            for i in range(nstarts):
                os.remove(progressfiles_multiprocess[i])

    else:
        for i in range(nstarts):
            if verbose > 5:
                print("\nStart {0}".format(i + 1))
            xvals = lb + (ub - lb) * zvals[i, :]
            if verbose > 20:
                print(str.join('\n', ['{0} : {1}'.format(s.x_names[i], xvals[j])
                                      for j, i in enumerate(multistart_xi)]))

            x_optimized[i, :], LL_optimized[i], niter_used[i], xp_all, LL_all = \
                sbfit_from_paramset(s, multistart_xi, xvals,
                                    maxiter_init=maxiter_init,
                                    maxiter_opt=maxiter_opt,
                                    print_every=print_every,
                                    fixed_params=fixed_params,
                                    verbose=verbose,
                                    save_progress=False)
            all_param_vals.append(xp_all)
            all_LL_vals.append(LL_all)

            if save_progress:
                with open(progressfile, 'wb') as fid:
                    pickle.dump(x_optimized, fid)
                    pickle.dump(LL_optimized, fid)
                    pickle.dump(niter_used, fid)
                    pickle.dump(all_param_vals, fid)
                    pickle.dump(all_LL_vals, fid)

    s.loglikelihood(x_optimized[np.argmax(LL_optimized), :])

    return x_optimized, LL_optimized, niter_used, all_param_vals, all_LL_vals


def sbfit_ps_wrapper(sbfit_input):
    x, LL, niter, xp_all, LL_all = sbfit_ps_dispatch(**sbfit_input)
    return dict(x=x, LL=LL, niter=niter, xp_all=xp_all, LL_all=LL_all)


def sbfit_ps_dispatch(s0=None, xi=None, xvals=None,
                      save_progress=False, progressfile='', index=-1,
                      **kwargs):

    if save_progress:
        x0 = s0.x.copy()
        x0[xi] = xvals
        with open(progressfile, 'wb') as fid:
            pickle.dump(False, fid)
            pickle.dump(index, fid)
            pickle.dump(x0, fid)
            pickle.dump(xi, fid)

    r = sbfit_from_paramset(s0, xi, xvals, **kwargs)

    if save_progress:
        with open(progressfile, 'wb') as fid:
            pickle.dump(True, fid)
            pickle.dump(index, fid)
            pickle.dump(x0, fid)
            pickle.dump(xi, fid)
            pickle.dump(r, fid)

    return r


def sbfit_from_paramset(solver0, xi, xvals,
                        maxiter_init=10,
                        maxiter_opt=5000,
                        print_every=10,
                        fixed_params=(),
                        verbose=100):
    '''
    Fix some params, do some iterations, then optimize over all (except masked).
    This function is called by sbfit_multistart.
    '''
    s = solver0.copy()

    xmask_global = ~np.array([n in fixed_params for n in s.x_names])
    assert s.isparam[~xmask_global].all(), "only params can be fixed"
    xpmask_global = xmask_global[s.isparam]
    assert xpmask_global[xi].all(), "cannot adjust fixed params"

    xpmask_otherparams = np.ones(s.nx, dtype=bool)
    xpmask_otherparams[xi] = False
    xpmask_otherparams = xpmask_otherparams[s.isparam]

    x = s.x.copy()
    x[xi] = xvals
    s.LL_profiled(x[s.isparam])

    # run a few iterations for the other params with the multistart params fixed
    xpmask_init = xpmask_otherparams & xpmask_global
    if xpmask_init.any() and maxiter_init > 0:
        if verbose > 10:
            print("Initializing extra parameters")
        _, niter1, xpall1, LL_all1 = s.solve_profile(verbose=verbose,
                                                     maxiter=maxiter_init,
                                                     print_every=print_every,
                                                     xpmask=xpmask_init)
    else:
        niter1, xpall1, = 0, np.zeros((0, s.isparam.sum()), dtype=float)
        LL_all1 = np.zeros(0, dtype=float)

    # now optimize over all non-fixed params
    if verbose > 10:
        print("Solving")
    _, niter2, xpall2, LL_all2 = s.solve_profile(verbose=verbose,
                                                 maxiter=maxiter_opt,
                                                 print_every=print_every,
                                                 xpmask=xpmask_global)
    LL = s.loglikelihood(s.x)
    niter = niter1 + niter2
    xpall = np.vstack((xpall1[:-1, :], xpall2))
    LL_all = np.append(LL_all1, LL_all2)

    return s.x, LL, niter, xpall, LL_all
