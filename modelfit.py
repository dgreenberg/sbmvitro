import warnings
import copy
import functools
import time
import numpy as np
import util
import scipy.optimize as opt
from scipy.optimize._numdiff import approx_derivative

mf_version = 0.11


class regressionmodel:

    def __init__(self,
                 data,
                 regressorfcn,
                 p0,
                 coefs=None,
                 regressorjac=None,
                 param_bounds=None,
                 coef_bounds=None,
                 parameter_names=None,
                 regressor_names=None,
                 data_types=None,
                 scale=1.0,
                 scale_name=None,
                 auto_scale=True,
                 noise_type=None,
                 modelname=None,
                 rel_step=None,
                 extra_inputs=None,  # extra inputs to regressorfcn and jac
                 xvals=None,  # abscissa for plotting etc.
                ):

        self.version = mf_version

        assert type(data) is np.ndarray, "data must be an ndarray"
        assert data.size > 0, "data cannot be empty"
        if data.ndim == 1:
            self.data = data.reshape(data.size, 1)
        else:
            assert data.ndim == 2, "data must have 1 or 2 dimensions"
            self.data = data
        self.datadim = self.data.shape[1]  # dimensionality of each measurement
        self.ndata = data.shape[0]

        if extra_inputs is None:
            extra_inputs = dict()
        self.extra_inputs = extra_inputs
        self.xvals = xvals

        self.nparams = p0.size

        # test / initialize the regressor function
        assert callable(regressorfcn), "regressor function must be callable"
        regressorfcnoutput = regressorfcn(p0, **extra_inputs)
        assert type(regressorfcnoutput) is tuple and \
            len(regressorfcnoutput) == 2, "regressorfcn must output a 2-tuple"
        self.regressorfcn = regressorfcn
        self.regressors = regressorfcnoutput[0]
        self.model_state = regressorfcnoutput[1]
        self.p = p0

        assert type(self.regressors) is np.ndarray and \
            len(self.regressors.shape) == 2 and \
            self.regressors.shape[0] == self.ndata, \
            "regressors must be a 2D ndarray with as many rows as data"
        self.nregressors = self.regressors.shape[1]

        self.regressorjac = regressorjac

        if regressorjac is not None:
            assert callable(regressorjac), "regressorjac must be callable"

        util.check_unique_string_list(
            parameter_names, n=self.nparams, illegal_chars='*')
        self.parameter_names = parameter_names

        util.check_unique_string_list(
            regressor_names, n=self.nregressors, illegal_chars='*')
        self.regressor_names = regressor_names

        if type(data_types) is str:
            data_types = [data_types]
        util.check_unique_string_list(data_types,
                                      n=self.datadim, illegal_chars='*')
        self.data_types = data_types

        if noise_type is not None:
            assert type(noise_type) is str, "noise_type must be a string"
            assert len(set(noise_type) & set('*')) == 0, "illegal character"
        self.noise_type = noise_type

        if scale_name is not None:
            assert type(scale_name) is str, "scale_name must be a string"
            assert len(set(scale_name) & set('*')) == 0, "illegal character"
        self.scale_name = scale_name

        self.scale = float(scale)
        self.auto_scale = auto_scale

        if modelname is not None:
            assert type(modelname) is str, "modelname must be a string"
            assert len(set(modelname) & set('*')) == 0, "illegal character"
        self.name = modelname

        self.param_bounds = util.check_bounds(param_bounds, (self.nparams,))
        self.coef_bounds = util.check_bounds(
            coef_bounds, (self.nregressors, self.datadim))

        if coefs is None:
            self.autoupdate_coefs()
        else:
            if coefs.ndim == 1:
                self.coefs = coefs.reshape(coefs.size, 1)
            else:
                assert coefs.ndim == 2, "coefs must have 1 or 2 dimensions"
                self.coefs = coefs
            assert coefs.shape[0] == self.nregressors, \
                "coefs must match the number of regressors"
            assert coefs.shape[1] == self.datadim, \
                "coefs must match the dimensionality of the data points"

        assert np.all((self.p >= self.param_bounds[0]) &
                      (self.p <= self.param_bounds[1])), \
            "initial p not within bounds"
        assert np.all((self.coefs >= self.coef_bounds[0]) &
                      (self.coefs <= self.coef_bounds[1])), \
            "initial coefs not within bounds"

        if rel_step is not None:
            rel_step = float(rel_step)
        self.rel_step = rel_step

    def set_parameters(self, p):
        if np.all(self.p == p):
            return
        self.p = p
        self.regressors, self.model_state = \
            self.regressorfcn(p, **self.extra_inputs)

    def autoupdate_coefs(self):
        data_scaled = self.data / self.scale
        if np.all(np.isneginf(self.coef_bounds[0]) &
                  np.isinf(self.coef_bounds[1])):  # linear regression
            self.coefs = np.linalg.lstsq(self.regressors, data_scaled)[0]
            return

        self.coefs = np.zeros((self.nregressors, self.datadim))
        for j in range(0, self.datadim):
            cb = (self.coef_bounds[0][:, j], self.coef_bounds[1][:, j])
            if (cb[0] == 0).all() and np.isposinf(cb[1]).all():
                self.coefs[:, j] = opt.nnls(
                    self.regressors,
                    data_scaled[:, j]
                )[0]
            else:
                self.coefs[:, j] = opt.lsq_linear(
                    self.regressors,
                    data_scaled[:, j],
                    cb
                ).x

    def gradients(self):
        '''returns derivatives of residuals with respect to params, coefs, and
        scale'''
        dr_dp = self.dregressors_dp()

        J_params = np.full((self.nparams, self.ndata, self.datadim), np.nan)
        for i in range(0, self.nparams):
            J_params[i, :, :] = np.dot(dr_dp[i, :, :], self.coefs * self.scale)

        # note this is the same for every column of coefs/data
        J_coefs = self.regressors * self.scale

        J_scale = np.dot(self.regressors, self.coefs)

        return (J_params, J_coefs, J_scale)

    def dregressors_dp(self, pmask=None):
        # calculate the derivatives of the regressor matrix with respect to
        # each of the parameters p
        if pmask is None:
            pmask = np.ones(self.nparams, dtype=bool)
        if self.regressorjac is None:
            dr_dp = self.dregressors_dp_fd(pmask=pmask)
        else:
            dr_dp = self.regressorjac(self.p, self.model_state,
                                      **self.extra_inputs)[pmask, :, :]
        return dr_dp

    def dregressors_dp_fd(self, pmask=None):
        '''
        the gradient of the regressors is not provided, so we have to calculate
        it numerically. note that rfr doesn't set model_state
        or p
        '''
        def rfr(pvals):
            ''' raveled regressor function '''
            return self.regressorfcn(pvals, **self.extra_inputs)[0].ravel()

        if pmask is None:
            pmask = np.ones(self.nparams, dtype=bool)
        p0sub, fsub = util.fix_inputs(self.p, rfr, pmask)
        subpbounds = (self.param_bounds[0][pmask], self.param_bounds[1][pmask])

        g = approx_derivative(  # copying p isn't necessary, but let's be safe
            fsub, p0sub,
            rel_step=self.rel_step, f0=self.regressors.ravel(),
            bounds=subpbounds)
        return g.T.reshape((pmask.sum(), self.ndata, self.nregressors))


class regressionsolver:

    def __init__(self,
                 models,
                 deepcopy=True,
                 optimize_variances=False,
                 init_unit_variances=True):
        if deepcopy:
            models = copy.deepcopy(models)
        models = as_modelslist(models)

        self.version = mf_version

        # total number of (in general multidimensional) data points over all
        # models
        self.ndata_total = 0
        # total number of observations, that is the summed dimensionality of
        # all data points
        self.nobs_total = 0

        self.nmodels = 0
        self.models = []

        self.nx = 0  # length of global vector x (params, coefs, scales, noise)
        # global vector of parameters, coefs and scales
        self.x = np.array([], dtype=np.float)
        self.iscoef = np.array([], dtype=bool)
        self.isscale = np.array([], dtype=bool)
        self.isvariance = np.array([], dtype=bool)
        self.isparam = np.array([], dtype=bool)
        self.bounds = (np.array([]), np.array([]))  # bounds on x
        self.x_names = []  # corresponds to self.x
        self.relevant_models = []  # list of models for each element of x

        self.x_index_params = []  # list over models of arrays over parameters

        self.nregressors_total = 0
        self.regressor_names = []
        self.regressor_index = []  # list over models of int arrays
        self.regressorlist_eachdatatype = [] # list over d.t.s of int arrays

        # coef_index_table[i, j] gives us an index into p for regressor i and
        # data_type j. unused coefs have a value of -1
        self.coef_index_table = np.zeros((0, 0), dtype=int)
        self.x_index_coefs = []  # list over models of 2D arrays over coefs

        self.ndatatypes = 0
        self.data_types = []  # list of all data types
        self.data_type_index = []  # list over models of int arrays
        self.models_each_datatype = []  # list over models of int arrays

        self.x_index_variance = np.array([], dtype=int)  # for each model

        self.nscales = 0
        self.scale_names = []
        self.scale_is_free = np.array([], dtype=bool)
        self.scale_index_eachmodel = np.array([], dtype=int)  # for each model

        # add the models
        for model in models:
            self.add_model(model)

        # determine which scales will be adjusted and which will be fixed etc.
        self.update_scale_info()

        # initialize variances to 1
        self.x[self.isvariance] = 1.0

        # initialize global parameters, coefs and scales
        self.make_models_consistent()  # optimizes scales as well

        # initialize the variances
        if optimize_variances:
            self.optimize_variances()
        elif not init_unit_variances:
            for xi in np.flatnonzero(self.isvariance):
                var = [m.noise_factor ** -2 for i, m in enumerate(self.models)
                       if self.x_index_variance[i] == xi and
                       'noise_factor' in dir(m)]
                if len(var) > 0:
                    self.x[xi] = np.mean(var)
                else:
                    self.x[xi] = 1.0

        # store a copy of initial parameters
        self.x0 = self.x.copy()
        self.prevx = None

        self.solveresult = None

    def copy(self):
        assert self.version == mf_version, "software has changed, cannot copy"
        s = regressionsolver(self.models,
                             deepcopy=True, optimize_variances=False)
        s.loglikelihood(self.x)  # updates model state etc. input is copied.
        return s

    def solve_profile(self,
                      verbose=100,
                      maxiter=np.inf,
                      print_every=10,
                      mintol=None,
                      tol_profile=0.0,
                      method=None,
                      xpmask=None):
        # does not adjust variances, but does use them
        if xpmask is None:
            xpmask = np.ones(self.isparam.sum(), dtype=bool)
        else:
            assert xpmask.shape == (self.isparam.sum(),), "invalid mask"

        ssqemode = np.all(self.x[self.isvariance] == 1.0)
        xp0 = self.x[self.isparam]
        xp_all, LL_all = [], []

        xi_maskedparams = np.flatnonzero(self.isparam)[xpmask]
        xpsub_lb = self.bounds[0][xi_maskedparams]
        xpsub_ub = self.bounds[1][xi_maskedparams]
        pbounds = [(self.bounds[0][xi], self.bounds[1][xi])
                   for xi in xi_maskedparams]  # for opt.minimize

        if mintol is None:
            mintol = 0.0  # np.finfo(float).eps
        minopts = dict(maxiter=maxiter)
        if method is None:
            method = 'L-BFGS-B'  # 'SLSQP'

        if method == 'L-BFGS-B':
            minopts['maxcor'] = np.minimum(20, xp0.size)
            minopts['maxls'] = 100


        def fcn(xpsub):
            xpsub = util.enforce_bounds(xpsub, xpsub_lb, xpsub_ub)
            xpfull = xp0.copy()
            xpfull[xpmask] = xpsub
            LL = self.LL_profiled(xpfull, tol_profile)
            if ssqemode:
                return self.ssqresiduals()
            else:
                return -LL

        def jac(xpsub):
            #  should be the same derivative in ssqemode, fcn differs by
            # a constant
            xpsub = util.enforce_bounds(xpsub, xpsub_lb, xpsub_ub)
            xpfull = xp0.copy()
            xpfull[xpmask] = xpsub  # fixme pass the mask all the way down
            return -self.LL_profiled_jac(xpfull, tol_profile)[xpmask]

        def callback(xpsub):
            xpsub  = util.enforce_bounds(xpsub, xpsub_lb, xpsub_ub)
            xpfull = xp0.copy()
            xpfull[xpmask] = xpsub
            xp_all.append(xpfull)
            LL = fcn(xpsub)  # should use solver/model caching
            LL_all.append(LL)
            callback.n += 1
            doprint = (callback.n % print_every) == 0 or callback.n == maxiter
            if doprint and verbose > 5:
                if ssqemode:
                    essq = self.ssqresiduals(varnorm=True)
                    print("{0:5}.  essq = {1:g}".format(callback.n, essq))
                else:
                    print("{0:5}.  LL = {1:g}".format(callback.n, LL))
        callback.n = 0

        t0 = time.time()

        xpsub0 = xp0[xpmask]

        xp_all.append(xp0.copy())
        LL_all.append(fcn(xpsub0))
        # keep restarting as long as possible. not sure why this is needed.
        # maybe the use of finite differences can lead to a bad hessian?
        niter_used = 2
        niter_total = 0
        r = None
        while niter_used > 1 and minopts['maxiter'] >= 1:
            r = opt.minimize(fcn,
                             xpsub0,
                             jac=jac,
                             method=method,
                             bounds=pbounds,
                             tol=mintol,
                             callback=callback,
                             options=minopts)
            niter_used = r.nit
            niter_total += niter_used
            minopts['maxiter'] -= niter_used
            xpsub0 = util.enforce_bounds(r.x, xpsub_lb, xpsub_ub)

        xp = xp0.copy()
        if r is not None:
            self.solveresult = r
            del self.solveresult.hess_inv
            xp[xpmask] = util.enforce_bounds(r.x, xpsub_lb, xpsub_ub)

        self.LL_profiled(xp)
        xp_all = np.array(xp_all)  # list of arrays -> 2D array

        t1 = time.time()
        if verbose > 3:
            print("Elapsed time: {0} seconds".format(t1 - t0))

        return r, niter_total, xp_all, LL_all

    def LL_profiled_jac(self,
                        xp,
                        tol=0.0):
        '''
        The argument here is that if the coefs are optimal, we can neglect
        the term derror_dcoef * dcoefs_dparams in the chain rule.
        If the coefs are not optimal but are at a bound, then they will move
        in the wrong direction (against the bound) for any infinitesmal change
        in the parameters, so again we can ignore that term.

        As a result, the gradient of the profiled log likelihood is simply the
        gradient of the likelihood w.r.t. the parameters
        '''
        x = self.x.copy()
        x[self.isparam] = xp
        self.assign_x(x)
        self.profile_coefs_and_scales(tol)
        return self.LLgrad_paramsonly(xp, xpmask=None)

    def profile_coefs_and_scales(self, tol):
        LL, LLprev = self.loglikelihood(self.x), -np.inf
        while LL - LLprev > tol:
            self.optimize_coefs_allmodels()
            self.optimize_scales_allmodels()
            LLprev = LL
            LL = self.loglikelihood(self.x)
        if LL < LLprev:
            return LLprev
        return LL

    def LL_profiled(self,
                    xp,
                    tol=0.0):
        # log likelihood for parameters, profiling over coefs/scales
        x = self.x.copy()
        x[self.isparam] = xp
        self.assign_x(x)
        return self.profile_coefs_and_scales(tol)

    def solve_alternating(self,
                          ftol=None,
                          adjust_variances=False,
                          verbose=100,
                          maxiter=np.inf,
                          print_every=10,
                          n_coef_updates=20,
                          xmask=None):

        if xmask is None:
            xmask = np.ones(self.nx, dtype=bool)

        if ftol is None:
            ftol = np.finfo(float).eps

        if self.isscale[xmask].sum() == 0:
            n_coef_updates = 1

        xpmask = xmask[self.isparam]

        lb = self.bounds[0]
        ub = self.bounds[1]
        lb_xp = self.bounds[0][self.isparam][xpmask]
        ub_xp = self.bounds[1][self.isparam][xpmask]

        lbfgsbopts = {'maxls': 50}

        # differnet format for bounds for lbfgs-b
        bounds_masked = [(lb[xi], ub[xi]) for xi in range(0, self.nx)
                         if self.isparam[xi] and xmask[xi]]

        '''
        we need to explicitly enforce the bounds due to some unknown rounding
        error inside the lbfgs-b code
        '''
        def fcn(x):
            x_bounded = np.minimum(np.maximum(x, lb), ub)
            return -self.loglikelihood(x_bounded)

        def jac(xparams):
            xparams_bounded = np.minimum(np.maximum(xparams, lb_xp), ub_xp)
            return -self.LLgrad_paramsonly(xparams_bounded, xpmask=xpmask)

        t0 = time.time()
        prevLL = self.loglikelihood(self.x)
        prevessq = self.ssqresiduals()
        self.prevx = self.x.copy()
        iteration = 0
        ssqemode = np.all(self.x[self.isvariance] == 1.0) and \
            not adjust_variances

        if verbose > 2:
            if ssqemode:
                print("Initial ssqe: {0}".format(prevessq))
            else:
                print("Initial LL: {0}".format(prevLL))

        while True:
            # optimize everything except parameters
            if adjust_variances:
                self.optimize_variances(xmask=xmask)
            for i in range(n_coef_updates):
                self.optimize_coefs_allmodels(xmask=xmask)
                self.optimize_scales_allmodels(xmask=xmask)

            LL = self.loglikelihood(self.x)
            essq = self.ssqresiduals()

            # determine tolerance
            '''
            if LL - prevLL > ftol:  # we definitely haven't converged
                ptol = (LL - prevLL) / 2.0
            else:
                ptol = ftol
            '''
            x0sub, fsub = util.fix_inputs(self.x,
                                          fcn,
                                          self.isparam & xmask)

            opts = lbfgsbopts.copy()
            if ssqemode:
                fastmode = (prevessq - essq) / prevessq > 0.01
            else:
                fastmode = (LL - prevLL) / np.abs(prevLL) > 0.01
            if fastmode:  # improved > 1% with coefs, scales and vars
                opts['maxiter'] = 10
                opts['maxls'] = 20
            elif LL - prevLL > ftol:
                opts['maxiter'] = 100

            self.solveresult = opt.minimize(fsub,
                                            x0=x0sub,
                                            jac=jac,
                                            bounds=bounds_masked,
                                            tol=ftol,
                                            method='L-BFGS-B',
                                            options=opts)
            del self.solveresult.hess_inv  # so we can deepcopy/pickle
            self.x[self.isparam & xmask] = self.solveresult.x
            LL = self.loglikelihood(self.x)
            essq = self.ssqresiduals()

            iteration += 1
            doprint = (iteration % print_every) == 0 or iteration == maxiter
            if verbose > 3 and doprint:
                nitused = self.solveresult.nit
                if ssqemode:
                    print("{0:5}.  ssqe = {1:10.3g}  diff = {2:8.3g} "
                          "lbfgs iterations = {3:4}".format(iteration,
                                                            essq,
                                                            prevessq - essq,
                                                            nitused))
                else:
                    print("{0:5}.  LL = {1:10.3g}  diff = {2:8.3g} "
                          "lbfgs iterations = {3:4}".format(iteration,
                                                            LL,
                                                            LL - prevLL,
                                                            nitused))

            if LL <= prevLL + ftol or iteration >= maxiter:
                if LL < prevLL:
                    warnings.warn("final iteration decreased LL, ignoring")
                    LL = prevLL
                    self.assign_x(self.prevx)
                break

            self.prevx = self.x.copy()
            prevLL = LL
            prevessq = essq

        if verbose > 2:
            if ssqemode:
                print("Final ssqe: {0}".format(essq))
            else:
                print("Final LL: {0}".format(LL))
        t1 = time.time()
        if verbose > 3:
            print("Elapsed time: {0} seconds".format(t1 - t0))

    def make_models_consistent(self):
        '''
        reassign shared parameters, coefs and scales so they are consistent
        across models
        '''
        self.make_scales_consistent()
        self.optimize_scales_allmodels()
        self.average_coefs()
        self.average_params()  # average regressor params and update regressors

    def make_scales_consistent(self):
        for s in range(0, self.nscales):
            if not self.scale_is_free[s]:
                continue

            # identify models using this scale
            mlist = np.flatnonzero(self.scale_index_eachmodel == s)

            # adjust coefs so scale is equal for all models using this scale
            base_model = mlist[0]
            base_scale = self.models[base_model].scale
            xi = self.x_index_modelscales[base_model]
            if xi == -1:
                continue  # this scale is fixed (e.g. equiv. class base scale)
            self.x[xi] = base_scale
            for m in mlist[1:]:
                model = self.models[m]
                model.coefs *= model.scale / base_scale
                model.scale = base_scale

    def average_params(self):
        '''average each parameter over models and update the regressors'''
        psum = np.zeros(self.nx, dtype=float)
        naveraged = np.zeros(self.nx, dtype=float)
        for m in range(0, self.nmodels):
            psum[self.x_index_params[m]] += self.models[m].p
            naveraged[self.x_index_params[m]] += 1

        ii = naveraged > 0
        assert np.all(
            ii == self.isparam), "failed to initialize some parameters"
        newpvals = psum[ii] / naveraged[ii]
        newpvals = np.clip(newpvals, self.bounds[0][ii], self.bounds[1][ii])
        self.x[ii] = newpvals
        self.assign_local_params()  # updates each model's regressors

    def average_coefs(self):
        '''average each coef over models'''
        psum = np.zeros(self.nx, dtype=float)
        naveraged = np.zeros(self.nx, dtype=float)
        for m in range(0, self.nmodels):
            xi = self.x_index_coefs[m].ravel()
            psum[xi] += self.models[m].coefs.ravel()
            naveraged[xi] += 1

        ii = naveraged > 0
        assert np.all(ii == self.iscoef), "failed to initialize some coefs"
        newcvals = psum[ii] / naveraged[ii]
        newcvals = np.clip(newcvals, self.bounds[0][ii], self.bounds[1][ii])
        self.x[ii] = newcvals
        self.assign_local_coefs()

    def optimize_scales_allmodels(self, xmask=None):
        '''optimize jointly over all scales'''
        for xi in range(0, self.nx):
            if not self.isscale[xi]:
                continue
            if (xmask is not None) and (not xmask[xi]):
                continue
            mlist = np.flatnonzero(self.x_index_modelscales == xi)
            nobs = np.sum(
                [self.models[m].ndata * self.models[m].datadim for m in mlist])
            A = np.zeros((nobs, 1), dtype=float)
            b = np.zeros((nobs, 1))
            offset = 0
            for m in mlist:
                model = self.models[m]
                sd = np.sqrt(self.x[self.x_index_variance[m]])

                datahat = np.dot(model.regressors, model.coefs).ravel()
                A[offset:offset + datahat.size, 0] = datahat / sd
                b[offset:offset + datahat.size, 0] = model.data.ravel() / sd
                offset += datahat.size
            assert offset == nobs, "incorrect matrix size"
            # note that scales are always positive but otherwise free

            lb = self.bounds[0][xi]
            ub = self.bounds[1][xi]

            self.x[xi] = np.maximum(
                lb, np.minimum(ub, np.linalg.lstsq(A, b)[0]))

        self.assign_local_scales()

    def find_data_column(self, data_type_index, model_index):
        '''
        Get the column for a given data type in a given model's data.
        We can't use np.searchsorted to find the index, since depending on the
        order in which the models were added to the solver, the integer values
        of self.data_type_index[model_index] may not be sorted.
        '''
        match = self.data_type_index[model_index] == data_type_index
        c = np.flatnonzero(match)
        assert c.size == 1, \
            "misssing or duplicate data type {0} in model {1}" \
            .format(data_type_index, model_index)
        return c[0]

    def optimize_coefs_allmodels(self, xmask=None):
        '''optimize jointly over all coefficients'''

        data_scaled, regs_scaled = [], []
        for i, m in enumerate(self.models):
            sd = np.sqrt(self.x[self.x_index_variance[i]])
            data_scaled.append(m.data / sd)
            regs_scaled.append(m.regressors * m.scale / sd)

        for dt in range(self.ndatatypes):
            # get a list of models for this datatype:
            mlist = self.models_each_datatype[dt]
            dtmodels = [self.models[i] for i in mlist]
            # get this data type's data column for each relevant model
            dtcolumns = [self.find_data_column(dt, i) for i in mlist]

            # sum up the number of data points for this data type
            nd = np.sum([m.ndata for m in dtmodels])
            # get a sorted list of regressor indices for this data type
            rlist = self.regressorlist_eachdatatype[dt]

            # get the indices into the global vector x for these coefficients
            xi_coefs = self.coef_index_table[rlist][:, dt]
            # get bounds on the coefs
            coef_bounds = (self.bounds[0][xi_coefs],
                           self.bounds[1][xi_coefs])

            # we're going to solve Ax = b. first we build A and b
            A = np.zeros((nd, len(rlist)), dtype=float)
            b = np.zeros(nd, dtype=float)
            offset = 0
            for j, i, m in zip(range(0, len(mlist)), mlist, dtmodels):
                sd = np.sqrt(self.x[self.x_index_variance[i]])
                Acols = np.searchsorted(rlist, self.regressor_index[i])
                A[offset:offset + m.ndata, Acols] = regs_scaled[i]
                b[offset:offset + m.ndata] = data_scaled[i][:, dtcolumns[j]]
                offset += m.ndata
            # apply the mask
            if (xmask is not None) and (not xmask[xi_coefs].all()):
                fixed_coefs = ~xmask[xi_coefs]
                if fixed_coefs.all():
                    continue  # nothing to update for this data type
                A = A[:, ~fixed_coefs]
                b -= np.dot(A[:, fixed_coefs], self.x[xi_coefs[fixed_coefs]])
            # now solve
            if np.all(np.isneginf(coef_bounds[0]) & np.isinf(coef_bounds[1])):
                coefs = np.linalg.lstsq(A, b)[0]
            elif np.all((coef_bounds[0] == 0) & (np.isinf(coef_bounds[1]))):
                coefs = opt.nnls(A, b)[0]
            else:
                coefs = opt.lsq_linear(A, b, coef_bounds).x
            if (xmask is not None) and (not xmask[xi_coefs].all()):
                xi_coefs_masked = xi_coefs[xmask[xi_coefs]]
                self.x[xi_coefs_masked] = coefs
            else:
                self.x[xi_coefs] = coefs

        self.assign_local_coefs()

    def update_scale_info(self):
        # this function analyzes the overlap in coefficients across models
        # and data scales, determines how many scale parameters we need in x
        # and assign them to each model as needed

        # make list of all indices in x vector for the coefs of each data scale
        x_index_coefs_eachscale = []
        models_each_scale = []
        for s in range(0, self.nscales):
            x_index_coefs_eachscale.append([])
            models_each_scale.append([])
        for m in range(0, self.nmodels):
            s = self.scale_index_eachmodel[m]
            models_each_scale[s].append(m)
            x_index_coefs_eachscale[s] = np.union1d(
                x_index_coefs_eachscale[s],
                self.x_index_coefs[m].ravel())

        # generate equivalance classes over scales
        # two scales are equivalent if they share a coef
        neqclasses = 0
        # equivalence class for each data scale
        self.eqclass_eachscale = np.full(self.nscales, -1, dtype=int)

        # groups of scale indices for each equivalence class:
        scale_indices_eacheqclass = []
        # all coef x indices for each equivalence class:
        x_index_coefs_eacheqclass = []

        for s in range(0, self.nscales):
            # indices into x for coefs of this scale's models
            xi = x_index_coefs_eachscale[s]
            for e in range(0, neqclasses):
                if np.intersect1d(xi, x_index_coefs_eacheqclass[e]).size > 0:
                    # we found an match, so assign this class and
                    # merge this scale's coef x indices into the equivalence
                    # class
                    x_index_coefs_eacheqclass[e] = np.union1d(
                        xi, x_index_coefs_eacheqclass[e])
                    self.eqclass_eachscale[s] = e
                    scale_indices_eacheqclass[e].append(s)
                    break
            if self.eqclass_eachscale[s] == -1:  # no match
                # create a new equivalence class
                self.eqclass_eachscale[s] = neqclasses
                x_index_coefs_eacheqclass.append(xi)
                scale_indices_eacheqclass.append([s])
                neqclasses += 1

        # scale index into x for each model. can be -1
        self.x_index_modelscales = np.full(self.nmodels, -1, dtype=int)
        for e in range(0, neqclasses):
            # assign a free scale parameter to the scales in the equivalence
            # class. make sure at least one is fixed since coefs are free
            # variables
            freescales = [s for s in scale_indices_eacheqclass[
                e] if self.scale_is_free[s]]
            if len(freescales) == len(scale_indices_eacheqclass[e]):
                freescales = freescales[1:]  # fix the first scale
            for s in freescales:
                self.add_x_element(self.scale_names[s], 'scale')
                xi = self.x_names.index(self.scale_names[s])
                for m in models_each_scale[s]:
                    self.x_index_modelscales[m] = xi
                    self.add_relevant_model(xi, m)
                    self.tighten_bounds(xi, 0.0, np.inf)

    def optimize_variances(self, xmask=None):
        nnoisetypes = sum(self.isvariance)
        ssqr = np.zeros(nnoisetypes)
        n = np.zeros(nnoisetypes)
        xi_variances = np.flatnonzero(self.isvariance)

        variance_index = np.searchsorted(xi_variances, self.x_index_variance)
        if xmask is None:
            vmask = np.ones(xi_variances.size, dtype=bool)
        else:
            vmask = xmask[self.isvariance]

        for i, m in enumerate(self.models):
            d = variance_index[i]
            if not vmask[d]:
                continue
            model = self.models[i]
            r = np.dot(model.regressors, model.coefs * model.scale) \
                - model.data
            ssqr[d] += (r ** 2).sum()
            n[d] += r.size

        assert (n[vmask] > 0).all, "failed to initialize some variances"
        V = ssqr[vmask] / n[vmask]
        assert not np.isnan(V).any(), "failed to initialize some variances"
        self.x[xi_variances[vmask]] = V
        # note that variances have no local assignment

    def add_x_element(self, name, variable_type):
        ''' add a new parameter to global list '''
        self.x_names.append(name)
        self.nx += 1
        self.bounds = (np.append(self.bounds[0], -np.inf),
                       np.append(self.bounds[1], np.inf))
        self.relevant_models.append(np.array([], dtype=int))
        self.x = np.append(self.x, np.nan)
        self.isparam = np.append(self.isparam, False)
        self.iscoef = np.append(self.iscoef, False)
        self.isscale = np.append(self.isscale, False)
        self.isvariance = np.append(self.isvariance, False)
        if variable_type == 'parameter':
            self.isparam[-1] = True
        elif variable_type == 'coef':
            self.iscoef[-1] = True
        elif variable_type == 'scale':
            self.isscale[-1] = True
        elif variable_type == 'variance':
            self.isvariance[-1] = True
        else:
            raise ValueError("Invalid variable type {0} for {1}"
                             .format(variable_type, name))

    def add_coef(self, m, i, j):
        '''
        i is local index of regressor
        j is local index of data dimension
        '''
        model = self.models[m]

        ri = self.regressor_index[m][i]
        dti = self.data_type_index[m][j]
        if ri not in self.regressorlist_eachdatatype[dti]:
            self.regressorlist_eachdatatype[dti] = \
                np.sort(np.append(self.regressorlist_eachdatatype[dti], ri))

        if self.coef_index_table[ri, dti] == -1:
            name = 'coef: (%s) / (%s)' % (self.data_types[
                dti], self.regressor_names[ri])
            self.add_x_element(name, 'coef')
            self.coef_index_table[ri, dti] = self.nx - 1

        xi = self.coef_index_table[ri, dti]
        self.x_index_coefs[m][i, j] = xi
        assert self.iscoef[xi], "inconsistent usage"

        self.add_relevant_model(xi, m)
        self.tighten_bounds(xi,
                            model.coef_bounds[0][i, j],
                            model.coef_bounds[1][i, j])

    def add_relevant_model(self, xi, m):
        if m not in self.relevant_models[xi]:
            self.relevant_models[xi] = np.append(self.relevant_models[xi], m)

    def add_noise_type(self, m):
        model = self.models[m]
        if model.noise_type is None:
            if model.name is None:
                name = '*model %d noise' % (m + 1)
            else:
                name = '*model %d (%s) noise' % (m + 1, model.name)
        else:
            name = model.noise_type
        if name not in self.x_names:
            self.add_x_element(name, 'variance')

        xi = self.x_names.index(name)
        assert self.isvariance[xi], "inconsistent usage"

        self.add_relevant_model(xi, m)
        self.tighten_bounds(xi, 0.0, np.inf)

        assert self.x_index_variance.size == m, "object out of sync"
        self.x_index_variance = np.append(self.x_index_variance, xi)

    def add_parameter(self, m, i):
        '''
        add one model's local parameter to the global fit.
        i is local index of parameter
        '''
        model = self.models[m]
        if model.parameter_names is None:
            if model.name is None:
                name = '*model %d parameter %d' % (m + 1, i + 1)
            else:
                name = '*model %d (%s) parameter %d' % (m +
                                                        1, model.name, i + 1)
        else:
            name = model.parameter_names[i]
            if name[-1] == '@':  # requires unique value for each model
                name = '*model {0} {1}'.format(m + 1, name[:-1])

        if name not in self.x_names:
            self.add_x_element(name, 'parameter')

        # find the index of this parameter in the global vector x by name
        xi = self.x_names.index(name)
        assert self.isparam[xi], "inconsistent usage"

        # store the global index
        self.x_index_params[m][i] = xi

        self.add_relevant_model(xi, m)
        self.tighten_bounds(xi,
                            model.param_bounds[0][i], model.param_bounds[1][i])

    def tighten_bounds(self, xi, lbval, ubval):
        lb = self.bounds[0]
        lb[xi] = max(lb[xi], lbval)
        ub = self.bounds[1]
        ub[xi] = min(ub[xi], ubval)
        self.bounds = (lb, ub)

    def add_data_type(self, m, i):
        '''register a model's data type for the solver'''
        model = self.models[m]
        if model.data_types is None:
            if model.name is None:
                name = '*model %d data type %d' % (m + 1, i + 1)
            else:
                name = '*model %d (%s) data type %d' % (m +
                                                        1, model.name, i + 1)
        else:
            name = model.data_types[i]

        if name not in self.data_types:
            self.data_types.append(name)
            self.ndatatypes += 1
            newvals = np.full((self.nregressors_total, 1), -1, dtype=int)
            self.coef_index_table = np.hstack((self.coef_index_table, newvals))
            # array of regressor indices for this data type is initially empty
            self.regressorlist_eachdatatype.append(np.array([], dtype=int))
            self.models_each_datatype.append(np.array([], dtype=int))

        dti = self.data_types.index(name)
        self.data_type_index[m][i] = dti
        self.models_each_datatype[dti] = \
            np.append(self.models_each_datatype[dti], m)

    def add_regressor(self, m, i):
        model = self.models[m]
        if model.regressor_names is None:
            if model.name is None:
                name = '*model %d regressor %d' % (m + 1, i + 1)
            else:
                name = '*model %d (%s) regressor %d' % (m +
                                                        1, model.name, i + 1)
        else:
            name = model.regressor_names[i]

        if name not in self.regressor_names:
            self.regressor_names.append(name)
            self.nregressors_total += 1
            newvals = np.full((1, self.ndatatypes), -1, dtype=int)
            self.coef_index_table = np.vstack((self.coef_index_table, newvals))

        self.regressor_index[m][i] = self.regressor_names.index(name)

    def add_scale(self, m):
        model = self.models[m]
        if model.scale_name is None:
            if model.name is None:
                name = '*model %d scale' % (m + 1)
            else:
                name = '*model %d (%s) scale' % (m + 1, model.name)
        else:
            name = model.scale_name
        if name not in self.scale_names:
            self.scale_is_free = np.append(
                self.scale_is_free, model.auto_scale)
            self.scale_names.append(name)
            self.nscales += 1

        si = self.scale_names.index(name)
        assert self.scale_index_eachmodel.size == m, "object out of sync"
        self.scale_index_eachmodel = np.append(self.scale_index_eachmodel, si)
        assert self.scale_is_free[si] == model.auto_scale, \
            "auto_scale must be the same for all models with the same scale"
        if not self.scale_is_free[si]:
            sv = [self.models[m].scale for m in (
                self.scale_index_eachmodel == si).nonzero()]
            assert np.all([t == sv[0] for t in sv]), \
                "fixed scale values are inconsistent across models"

    def add_model(self, model):
        '''add a model to the solver'''
        n = self.nmodels  # index for the model we're now adding

        # add the model to the solver
        self.models.append(model)
        self.nmodels += 1
        self.ndata_total += model.ndata
        self.nobs_total += model.ndata * model.datadim

        # register the model's data types
        self.data_type_index.append(np.zeros(model.datadim, dtype=int))
        for i in range(0, model.datadim):
            self.add_data_type(n, i)

        # register the model's regressor types
        self.regressor_index.append(np.zeros(model.nregressors, dtype=int))
        for i in range(0, model.nregressors):
            self.add_regressor(n, i)

        # add local parameters for this model, creating new globals as needed
        # self.x_index_params[n] will store indices into global vector for
        # each of this model's parameters
        self.x_index_params.append(np.zeros(model.nparams, dtype=int))
        for i in range(0, model.nparams):
            self.add_parameter(n, i)

        # add coefs for this model, creating new globals as needed
        self.x_index_coefs.append(np.zeros_like(model.coefs, dtype=int))
        for i in range(0, model.nregressors):
            for j in range(0, model.datadim):
                self.add_coef(n, i, j)

        self.add_noise_type(n)

        self.add_scale(n)

    def assign_local_params(self):
        # set_parameters updates the regressors as well
        for m in range(0, self.nmodels):
            self.models[m].set_parameters(self.x[self.x_index_params[m]])

    def assign_local_coefs(self):
        for m in range(0, self.nmodels):
            model = self.models[m]
            newcoefs = self.x[self.x_index_coefs[m].ravel()]
            model.coefs = newcoefs.reshape(model.coefs.shape)

    def assign_local_scales(self):
        for m in range(0, self.nmodels):
            if self.x_index_modelscales[m] != -1:
                self.models[m].scale = self.x[self.x_index_modelscales[m]]

    def assign_locals(self):
        ''' use the global paramter vector pvec to assign local paramters and
        coefs to each model '''
        self.assign_local_params()
        self.assign_local_coefs()
        self.assign_local_scales()

    def global_residuals(self, x):
        # residuals are in order of models, then the residuals for each model
        # occur row by row
        self.x = x
        self.assign_locals()
        r = np.full(self.nobs_total, np.nan)
        offset = 0
        for m in range(0, self.nmodels):
            model = self.models[m]
            nd = model.ndata * model.datadim
            r_raw = np.dot(model.regressors, model.coefs *
                           model.scale) - model.data

            sd = np.sqrt(self.x[self.x_index_variance[m]])

            r[offset:offset + nd] = r_raw.ravel() / sd
            offset += nd

        return r

    def global_jacobian_fd(self, x, rel_step=None):
        '''
        finite difference version of global_jacobian. for testing purposes only
        '''
        x0 = self.x.copy()
        J = approx_derivative(
            self.global_residuals, x,
            rel_step=rel_step,
            bounds=self.bounds
        )

        self.assign_x(x0)

        return J[:, ~self.isvariance]

    def global_jacobian(self, x):
        '''
        derivative of residuals with respect to self.x, which includes
        parameters, coefs and scales
        residuals are in order of models, then the residuals for each model
        occur row by row
        '''
        self.x = x
        self.assign_locals()
        J = np.zeros((self.nx, self.nobs_total), dtype=float)
        offset = 0
        for m in range(0, self.nmodels):
            model = self.models[m]
            nd = model.ndata * model.datadim

            sd = np.sqrt(self.x[self.x_index_variance[m]])

            J_params, J_coefs, J_scale = model.gradients()
            J_coefs = J_coefs.T

            # all this model's residuals can depend on all its params
            J[self.x_index_params[m], offset:offset + nd] = \
                J_params.reshape(model.nparams, -1) / sd

            # each column of residuals depends on a single column of coefs
            for d in range(0, model.datadim):
                '''
                we need to assign derivatives for a column of residuals.
                we use a stride of model.datadim since residuals are in
                row order.
                '''
                s = offset + d
                J[self.x_index_coefs[m][:, d], s:offset + nd:model.datadim] = \
                    J_coefs / sd

            # if this model's scale is an element of x, include the relevant
            # column in the Jacobian
            if self.x_index_modelscales[m] != -1:
                # all residuals depend on the scale
                J[self.x_index_modelscales[m], offset:offset + nd] = \
                    J_scale.reshape(1, -1) / sd

            offset += nd

        return J[~self.isvariance, :].T

    def loglikelihood(self, x):
        self.assign_x(x)

        L = 0.0
        for m, model in enumerate(self.models):
            # residuals
            r = np.dot(model.regressors, model.coefs * model.scale) - \
                model.data
            # variance
            V = self.x[self.x_index_variance[m]]

            L -= (r ** 2).sum() / V + np.log(2 * np.pi * V) * r.size

        return L / 2.0

    def LLgrad_fd(self, x, rel_step=None):
        '''
        for testing purposes.
        rescales elements of x corresponding to variances
        so that the step size will not be too large.
        '''
        self.assign_x(x)
        x0 = self.x.copy()
        slope = np.ones_like(x0)

        slope[self.isvariance] = x0[self.isvariance]

        lb = (self.bounds[0] - x0) / slope
        ub = (self.bounds[1] - x0) / slope
        bounds_v = (lb, ub)

        def q(v):
            return self.loglikelihood(x0 + slope * v)

        g = approx_derivative(
            q,
            np.zeros_like(x0),
            rel_step=rel_step,
            bounds=bounds_v
        )

        self.assign_x(x0)  # undo any changes to x

        return g / slope  # chain rule

    # def LLgrad_paramsonly_fd(self, xparams, rel_step=None):

    def LLgrad_paramsonly(self, xparams, xpmask=None):
        if xpmask is None:
            xpmask = np.ones(self.isparam.sum(), dtype=bool)
        xi_params_masked = np.flatnonzero(self.isparam)[xpmask]
        self.x[xi_params_masked] = xparams
        self.assign_x(self.x)

        g = np.zeros(self.nx, dtype=float)
        for i, m in enumerate(self.models):
            pmask = np.array([j in xi_params_masked
                              for j in self.x_index_params[i]])
            npm = pmask.sum()
            if npm == 0:
                continue

            # variance
            V = self.x[self.x_index_variance[i]]
            # residuals
            r = np.dot(m.regressors, m.coefs * m.scale) - m.data
            rn = r.reshape(-1) / V

            cs = m.coefs * m.scale
            dregressors_dp = m.dregressors_dp(pmask=pmask)

            dresiduals_dp = np.empty((npm, m.ndata, m.datadim))
            for j in range(0, npm):
                dresiduals_dp[j, :, :] = np.dot(dregressors_dp[j, :, :], cs)

            g[self.x_index_params[i][pmask]] -= \
                np.dot(dresiduals_dp.reshape(npm, -1), rn)

        return g[self.isparam][xpmask]

    def LLgrad(self, x):
        self.assign_x(x)

        g = np.zeros(self.nx, dtype=float)
        for i, m in enumerate(self.models):
            # residuals
            r = np.dot(m.regressors, m.coefs * m.scale) - \
                m.data
            rssq = (r ** 2).sum()
            n = r.size
            # variance
            V = self.x[self.x_index_variance[i]]

            # get derivatives of invidual residual terms with respect to
            # params, coefs and scales
            J_params, J_coefs, J_scale = m.gradients()

            g[self.x_index_params[i]] -= \
                np.dot(J_params.reshape(m.nparams, -1), r.reshape(-1)) / V

            for d in range(0, m.datadim):
                g[self.x_index_coefs[i][:, d]] -= np.dot(r[:, d], J_coefs) / V

            # if this model's scale is an element of x, include the relevant
            # gradient terms
            if self.x_index_modelscales[i] != -1:
                g[self.x_index_modelscales[i]] -= (J_scale * r).sum() / V

            dLL_dV = -0.5 * (n / V - rssq / (V ** 2))
            g[self.x_index_variance[i]] += dLL_dV

        return g

    def assign_x(self, x):
        self.x = x.copy()
        self.assign_locals()

    def ssqresiduals(self, varnorm=False):
        ssqe = 0.0
        for i, m in enumerate(self.models):
            residuals = np.dot(m.regressors, m.coefs * m.scale) - m.data
            if varnorm:
                var = self.x[self.x_index_variance[i]]
            else:
                var = 1.0
            ssqe += (residuals ** 2).sum() / var
        return ssqe

    def print_params(self, precision=5):
        if self.nx == 0:
            return
        print(self.param_description(precision=precision))

    def param_description(self, precision=5):
        '''print parameter values, bounds and indices'''
        if self.nx == 0:
            return ''

        xi_p = [i for i in range(self.nx) if self.isparam[i]]

        nchars_x = np.max([len('{0:.{p}}'.format(self.x[i],
                                                 p=precision)) for i in xi_p])
        nchars_x = np.maximum(nchars_x, len("value"))
        nchars_n = np.max([len('{0}'.format(self.x_names[i])) for i in xi_p])
        nchars_n = np.maximum(nchars_n, len("name"))
        nchars_lb = np.max([len('{0:.{p}}'.format(self.bounds[0][i],
                                                  p=precision)) for i in xi_p])
        nchars_lb = np.maximum(nchars_lb, len("LB"))
        nchars_ub = np.max([len('{0:.{p}}'.format(self.bounds[1][i],
                                                  p=precision)) for i in xi_p])
        nchars_ub = np.maximum(nchars_ub, len("UB"))
        nchars_i = np.max([len('{0}'.format(i)) for i in xi_p])
        nchars_i = np.maximum(nchars_i, len("index"))

        labels = "{x:>{nchars_x}}  {n:<{nchars_n}}  {lb:>{nchars_lb}}" \
                 "  {ub:>{nchars_ub}}  {xi:>{nchars_i}}" \
                 .format(x="value", nchars_x=nchars_x,
                         n="name", nchars_n=nchars_n,
                         lb="LB", nchars_lb=nchars_lb,
                         ub="UB", nchars_ub=nchars_ub,
                         xi="index", nchars_i=nchars_i)
        pstr = ["{x:{nchars_x}.{p}}  {n:{nchars_n}}  {lb:{nchars_lb}.{p}}"
                "  {ub:{nchars_ub}.{p}}  {xi:{nchars_i}}"
                .format(x=self.x[i], nchars_x=nchars_x,
                        n=self.x_names[i], nchars_n=nchars_n,
                        lb=self.bounds[0][i], nchars_lb=nchars_lb,
                        ub=self.bounds[1][i], nchars_ub=nchars_ub,
                        xi=i, nchars_i=nchars_i,
                        p=precision)
                for i in xi_p]

        return labels + '\n\n' + '\n'.join(pstr)


def estimate_residual_scales(models, experiments,
                             maxiter=5000,
                             verbose=100):
    etypes = list(set([e['type'] for e in experiments]))
    residual_scales = dict()
    for etype in etypes:
        etype_models = [m for i, m in enumerate(models)
                        if experiments[i]['type'] == etype]
        etype_solver = regressionsolver(etype_models,
                                        optimize_variances=False)
        etype_solver.solve_profile(verbose=verbose,
                                   maxiter=maxiter)
        ssqe_models = \
            [((m.data - np.dot(m.regressors, m.coefs * m.scale))**2).sum()
             for m in etype_solver.models]
        ndatapts = np.sum([m.data.size for m in etype_solver.models])
        residual_scales[etype] = np.sqrt(np.sum(ssqe_models) / ndatapts)

    return residual_scales


def reweight_data(models, experiments, deepcopy=True, residual_scales=dict()):
    '''
    rescale model data to account for differences in units and number of
    observations across experiment types so that each type of experiment
    contributes roughly equally to a model fit
    '''
    if deepcopy:
        models = copy.deepcopy(models)
    residual_scales = copy.deepcopy(residual_scales)
    # get a unique list of experiment types:
    etypes = list(set([e['type'] for e in experiments]))

    # get the sum of absolute values for each experiment type:
    sum_abs_data = np.zeros(len(etypes))
    n_observations = np.zeros(len(etypes))
    for i, e in enumerate(experiments):
        etind = etypes.index(e['type'])
        sum_abs_data[etind] += np.abs(models[i].data).sum()
        n_observations[etind] += models[i].data.size
    # approximate residual scales as one tenth of data scales as needed
    for etind, et in enumerate(etypes):
        if et not in residual_scales.keys():
            residual_scales[et] = \
                0.1 * sum_abs_data[etind] / n_observations[etind]
    '''
    we want to normalize by dividing by some factor f.
    if the residual scale is d, then  before normalization we will have:
    have e = sum(residuals**2) = nobs * d**2
    and after normalization:
    e = nobs * d**2 / f**2
    so to get the same weighting for data types, we should set f for each data
    type to:
    f = sqrt(nobs) * d
    '''
    normfacs = np.array([np.sqrt(n_observations[etind]) * residual_scales[et]
                         for etind, et in enumerate(etypes)])
    # adjust the global scale of normfacs so the data scale will be about 1.0:
    mean_normedval = (sum_abs_data / normfacs).sum() / n_observations.sum()
    normfacs /= mean_normedval

    for i, e in enumerate(experiments):
        etind = etypes.index(e['type'])
        models[i].data /= normfacs[etind]  # rescale data
        models[i].coefs /= normfacs[etind]  # rescale coefs to match
        models[i].normalization_factor = normfacs[etind]

    return models


def as_modelslist(models):
    if type(models) is not list:
        models = [models]
    return models
'''
    if type(models) is regressionmodel:
        models = [models]
    else:
        assert type(models) is list, \
            "models must be a regressionmodel or a list of them"
        assert np.all([type(m) is regressionmodel for m in models]), \
            "models must be a regressionmodel or a list of them"
    return models
'''


def simulate_data(regressorfcn, p, coefs, extra_inputs=dict(),
                  scale=1.0, sigma=0.0):
    '''
    simulate data from a regression model
    '''
    r = regressorfcn(p, **extra_inputs)
    if type(r) is tuple:
        r = r[0]
    data = np.dot(r, coefs * scale)
    data += sigma * np.random.randn(r.shape[0], coefs.shape[1])
    return data


def retrieve_params(model, parameter_names):
    v = np.full(len(parameter_names), np.nan)
    for i, name in enumerate(parameter_names):
        if name in model.parameter_names:
            j = model.parameter_names.index(name)
            v[i] = model.p[j]
    return v


def assign_params(model, parameter_names, v):
    any_assigned = False
    p = model.p.copy()
    for i, name in enumerate(parameter_names):
        if name in model.parameter_names:
            j = model.parameter_names.index(name)
            p[j] = v[i]
            any_assigned = True
    if any_assigned:
        model.set_parameters(p)


def getsolverparams(s):
    # params we can assign. e.g. betas, contaminations, purities etc.
    param_names = [name for i, name in enumerate(s.x_names)
                   if s.isparam[i] and name[0] != '*']
    param_vals = np.array([s.x[s.x_names.index(name)]
                           for name in param_names])
    return (param_names, param_vals)


def assign_coefs_fromsolver(m, s):
    for ri, rn in enumerate(m.regressor_names):
        if rn not in s.regressor_names:
            continue
        ri_s = s.regressor_names.index(rn)
        for dti, dt in enumerate(m.data_types):
            if dt not in s.data_types:
                continue
            dt_s = s.data_types.index(dt)
            # assign the appropriate value
            m.coefs[ri, dti] = \
                s.x[s.coef_index_table[ri_s, dt_s]]


def collapse_regressors(m, regressor_groups, optimize_coefs=True):
    '''
    merge groups of regressors in a model to they must share the same
    coefficients
    '''
    assert type(regressor_groups) is list, "regressor_groups must be a list"
    all_members = functools.reduce(np.union1d, regressor_groups)
    assert (all_members == np.arange(m.nregressors)).all(), "every regressor" \
        "must be in exactly one group"
    assert functools.reduce(np.add, [len(g) for g in regressor_groups]) == \
        m.nregressors, "every regressor must be in exactly one group"
    regressor_groups = copy.deepcopy(regressor_groups)
    ngroups = len(regressor_groups)
    m = copy.deepcopy(m)

    def rfcn_collapsed(p, **extra_inputs):
        r, model_state = m.regressorfcn(p, **extra_inputs)
        regressors = util.collapse_columns(r, regressor_groups)
        return regressors, model_state

    if m.regressorjac is None:
        rjac_collapsed = None
    else:
        def rjac_collapsed(p, model_state, **extra_inputs):
            dr_dp = m.regressorjac(p, model_state, **extra_inputs)
            J = np.zeros((m.nparams, m.ndata, ngroups), dtype=float)
            for j, g in enumerate(regressor_groups):
                for i in g:
                    J[:, :, j] += dr_dp[:, :, i]
            return J

    regressor_names = []
    coefs = np.full((ngroups, m.datadim), np.nan)
    lb = np.full((ngroups, m.datadim), -np.inf)
    ub = np.full((ngroups, m.datadim), np.inf)
    for j, g in enumerate(regressor_groups):
        regressor_names.append(' + '.join([m.regressor_names[i] for i in g]))
        coefs_sum = np.zeros(m.datadim, dtype=float)
        for i in g:
            lb[j, :] = np.maximum(lb[j, :], m.coef_bounds[0][i, :])
            ub[j, :] = np.minimum(ub[j, :], m.coef_bounds[1][i, :])
            coefs_sum += m.coefs[i, :]
        coefs[j, :] = coefs_sum / len(g)
    assert (lb < ub).all(), "cannot collapse, inconsistent coefficient bounds"

    modelname = m.name
    if modelname is not None:
        modelname += ' (collapsed)'

    mc = regressionmodel(data=m.data,
                         regressorfcn=rfcn_collapsed,
                         p0=m.p,
                         coefs=coefs,
                         regressorjac=rjac_collapsed,
                         param_bounds=m.param_bounds,
                         coef_bounds=(lb, ub),
                         parameter_names=m.parameter_names,
                         regressor_names=regressor_names,
                         data_types=m.data_types,
                         scale=m.scale,
                         scale_name=m.scale_name,
                         auto_scale=m.auto_scale,
                         noise_type=m.noise_type,
                         modelname=modelname,
                         rel_step=m.rel_step,
                         extra_inputs=m.extra_inputs)

    if optimize_coefs:
        mc.autoupdate_coefs()

    extrafields = ['noise_factor', 'normalization_factor',
                   'concentration_unit', 'time', 'xvals']
    util.copyfields(m, mc, extrafields, dowarn=False)
    mc.regressor_groups = regressor_groups

    return mc
