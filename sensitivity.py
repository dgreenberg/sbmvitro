import modelfit as mf
import numpy as np
import names
import pickle
from multiprocessing import Pool, cpu_count
import time


def sbfit_sensitivity(s, nbindingsteps, fixed_params=[], **opts):
    allnames = [pname for pname in names.binding_param_names(nbindingsteps)
                if pname not in fixed_params]
    sensitivity_xi = np.array([], dtype=int)
    for pname in allnames:
        if pname in s.x_names and s.isparam[s.x_names.index(pname)]:
            sensitivity_xi = np.append(sensitivity_xi, s.x_names.index(pname))
    assert(sensitivity_xi.size > 0), "no valid params for sensitivity"
    return sensitivity_analysis(s, sensitivity_xi,
                                fixed_params=fixed_params, **opts)


def sensitivity_analysis(s,
                         sensitivity_xi,
                         multiprocess=True,
                         nworkers=None,
                         fixed_params=[],
                         save_progress=False,
                         progressfile='sensitivity progress.pickle',
                         max_changefrac=0.1,
                         nsteps=10,
                         verbose=100,
                         maxiter_opt=5000):
    xpmask = ~np.array([n in fixed_params
                       for i, n in enumerate(s.x_names)
                       if s.isparam[i]])
    n_sensitivity_params = sensitivity_xi.size
    xorig = s.x.copy()
    LL0 = s.LL_profiled(s.x[s.isparam])
    x0 = s.x.copy()

    LL_perturbed = np.full((n_sensitivity_params, 2 * nsteps + 1), np.nan)
    LL_perturbed[:, nsteps] = LL0
    perturbed_param_vals = \
        np.full((n_sensitivity_params, 2 * nsteps + 1), np.nan)
    perturbed_param_vals[:, nsteps] = x0[sensitivity_xi]
    perturbed_x = np.full((n_sensitivity_params, 2 * nsteps + 1, s.nx), np.nan)
    for i in range(n_sensitivity_params):
        perturbed_x[i, nsteps, :] = x0.copy()

    if multiprocess:  # fixme: the code in this branch could be more compact
        sc_inputs = []
        for xi in sensitivity_xi:
            for d in [-1, 1]:
                sc_inputs.append(dict(s0=s, xi=xi, xpmask=xpmask, direction=d,
                                      maxiter=maxiter_opt,
                                      max_changefrac=max_changefrac,
                                      nsteps=nsteps, verbose=0))

        if nworkers is None:
            nworkers = np.minimum(cpu_count(), len(sc_inputs))
        t0 = time.time()
        with Pool(nworkers) as p:
            results = p.map(sc_wrapper, sc_inputs)
        t1 = time.time()
        if verbose > 3:
            print("Elapsed (wall clock) time: {0} seconds".format(t1 - t0))

        for u, r in enumerate(results):
            i = np.flatnonzero(sensitivity_xi == sc_inputs[u]['xi'])
            d = sc_inputs[u]['direction']
            jj = nsteps + d * np.arange(1, nsteps + 1)
            LL_perturbed[i, jj] = r['LL']
            perturbed_param_vals[i, jj] = r['pv']
            perturbed_x[i, jj, :] = r['px']

    else:
        for i, xi in enumerate(sensitivity_xi):
            for d in [-1, 1]:  # perturbation directions
                nextLL, nextpv, nextpx = \
                    sensitivity_curve(s, xi, xpmask=xpmask, direction=d,
                                      maxiter=maxiter_opt,
                                      max_changefrac=max_changefrac,
                                      nsteps=nsteps, verbose=verbose)

                jj = nsteps + d * np.arange(1, nsteps + 1)
                LL_perturbed[i, jj] = nextLL
                perturbed_param_vals[i, jj] = nextpv
                perturbed_x[i, jj, :] = nextpx

                if save_progress:
                    with open(progressfile, 'wb') as fid:
                        pickle.dump(LL_perturbed, fid)
                        pickle.dump(perturbed_param_vals, fid)
                        pickle.dump(perturbed_x, fid)

    s.loglikelihood(xorig)  # reset to original params
    # fixme return number of iterations used for each optimization
    return LL_perturbed, perturbed_param_vals, perturbed_x


def sc_wrapper(sc_input):
    LL, pv, px = sc_dispatch(**sc_input)
    return dict(LL=LL, pv=pv, px=px)


def sc_dispatch(s0=None, xi=None, **kwargs):
    return sensitivity_curve(s0, xi, **kwargs)


def sensitivity_curve(s0,
                      xi,
                      xpmask=None,
                      direction=1,  # 1 or -1
                      maxiter=5000,
                      max_changefrac=0.1,
                      nsteps=10,
                      verbose=100):
    assert direction == 1 or direction == -1, "direction must be +/-1"
    if xpmask is None:
        xpmask = np.ones(s0.isparam.sum(), dtype=bool)
    else:
        xpmask = xpmask.copy()
    assert xpmask[xi], "sensitivity parameter is masked out"
    xpmask[xi] = False  # fix after perturbing
    s = s0.copy()
    pval0 = s.x[xi]
    s.loglikelihood(s0.x.copy())

    LL_perturbed = np.full(nsteps, np.nan)
    perturbed_param_vals = np.full(nsteps, np.nan)
    perturbed_x = np.full((nsteps, s.nx), np.nan)

    s.LL_profiled(s.x[s.isparam])
    for j in range(nsteps):
        r = max_changefrac * (j + 1) / nsteps
        pval = pval0 * (1.0 + r * direction)
        if (pval < s.bounds[0][xi] or pval > s.bounds[1][xi]):
            break
        if verbose > 10:
            print("{0} = {1}\n".format(s.x_names[xi], pval))
        x = s.x.copy()
        x[xi] = pval
        s.LL_profiled(x[s.isparam])
        s.solve_profile(verbose=verbose,
                        maxiter=maxiter,
                        xpmask=xpmask)
        LL_perturbed[j] = s.loglikelihood(s.x)
        perturbed_param_vals[j] = pval
        perturbed_x[j, :] = s.x.copy()

    return LL_perturbed, perturbed_param_vals, perturbed_x
