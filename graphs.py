import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import ipynb_tools as nbt
import kinetics as ki
import equilibrium as eq
import util
import names


bufferfac = 0.03  # extra ylim space around min/max values on some graphs

binding_state_colors = ['b', 'g', 'r', 'c', 'm', 'y',
                        'k', 'orange', 'chartreuse', 'thistle']


def bscolor(i):
    return binding_state_colors[i % len(binding_state_colors)]


def plot_for_each_binding_state(x, y, semilogx=False):
    h = []
    for i in range(y.shape[1]):
        if semilogx:
            h += plt.semilogx(x, y[:, i], color=bscolor(i))
        else:
            h += plt.plot(x, y[:, i], color=bscolor(i))
    return h


def plot_bindingstatesfunc(yraw,
                           yscale=1.0,
                           ylabeltext=None,
                           legendtext=None,
                           logscale=False):
    '''
    Plot a function defined on binding states, with appropriate x-tick labels
    '''
    if type(yraw) == tuple:
        y = np.vstack(yraw).transpose()
    else:
        y = yraw.copy()
    n = y.shape[0] - 1  # max. number of bound ligands

    y *= yscale

    if logscale:
        h = plt.semilogy(np.arange(0, n + 1), y, 'o-')
    else:
        h = plt.plot(np.arange(0, n + 1), y, 'o-')

    yrange = y.max() - y.min()
    if yrange != 0:
        plt.ylim((y.min() - yrange * bufferfac, y.max() + yrange * bufferfac))

    plt.xlabel('Binding state')
    plt.xticks(np.arange(0, n + 1), [bsname(j) for j in range(0, n + 1)])
    if ylabeltext is not None:
        nbt.ylabelh(ylabeltext)

    if legendtext is not None:
        plt.legend(h, legendtext, loc='best')


def plot_fstates(fstates, legendtext=None, logscale=False):
    plot_bindingstatesfunc(
        fstates,
        yscale=1e-6,
        ylabeltext='Binding state\nfluorescence ($\\mu M^{-1}$)',
        legendtext=legendtext,
        logscale=logscale)


def bsname(j):  # fixme replace with names.bindingstatenames
    '''
    Name of binding state for legends, tick labels, etc.
    '''
    if j == 0:
        return '$[M]$'
    else:
        return '$[L_{%d}M]$' % j


def plot_fdecomp(tobs, y, fstates, fdata=np.zeros(0)):
    '''
    plot contributions of each binding state to a fluorescence time series
    '''
    n = fstates.size - 1  # max. number of bound ligands
    c = y[0:n + 1, :].transpose()  # binding state concentrations
    f_decomp = c * fstates
    fmodel = f_decomp.sum(axis=1)
    fall = np.hstack((fmodel.reshape(-1, 1), f_decomp))
    legendtext = ['$F$ (model)'] + ['$F_{%d}$%s (model)' %
                                    (i, bsname(i)) for i in range(0, n + 1)]
    if fdata.size > 0:
        fall = np.hstack((fdata.reshape(-1, 1), fall))
        legendtext = ['$F$ (data)'] + legendtext

    hf = []
    for i in range(fstates.size):
        hf += plt.plot(tobs, fall, color=bscolor(i))

    plt.xlabel('Time (sec)')
    nbt.ylabelh('Fluorescence (AU)')
    plt.legend(hf, legendtext, loc='best')
    plt.setp(hf[0], linewidth=2)
    if fdata.size > 0:
        plt.setp(hf[1], linewidth=2)

    figh = plt.gcf()
    plt.show()
    return figh


def check_stoppedflow_param_sizes(b, u, fstates):
    assert \
        len(b.shape) == 1 and \
        len(u.shape) == 1 and \
        len(fstates.shape) == 1 \
        and b.size == u.size and \
        fstates.size == b.size + 1, \
        "b and u must be the same size, and fstates must be one element longer"


def plot_stoppedflow_params(braw, uraw, fstates, groupnames=None):
    if type(braw) == tuple:
        assert type(uraw) == tuple and type(fstates) == tuple, "invalid input"
        assert len(braw) == len(uraw) == len(fstates), "invalid input"
        for ii in range(0, len(uraw)):
            check_stoppedflow_param_sizes(braw[ii], uraw[ii], fstates[ii])
        b = np.vstack(braw).transpose()
        u = np.vstack(uraw).transpose()
    else:
        b, u = braw, uraw
        check_stoppedflow_param_sizes(b, u, fstates)

    n = b.shape[0]

    plt.subplot(4, 1, 1)
    h = plt.semilogy(np.arange(1, n + 1), b, 'o-')
    plt.xticks(np.arange(1, n + 1))
    nbt.ylabelh('On rates ($M^{-1}s^{-1}$)')
    if groupnames is not None:
        plt.legend(h, [
            '$b_j$ (%s)' % groupnames[0],
            '$b_j$ (%s)' % groupnames[1]
        ], loc='best')

    plt.subplot(4, 1, 2)
    h = plt.plot(np.arange(1, n + 1), u, 'o-')
    plt.xticks(np.arange(1, n + 1))
    nbt.ylabelh('Off rates ($s^{-1}$)')
    if groupnames is not None:
        plt.legend(h, [
            '$u_j$ (%s)' % groupnames[0],
            '$u_j$ (%s)' % groupnames[1]
        ], loc='best')

    plt.subplot(4, 1, 3)
    h = plt.plot(np.arange(1, n + 1), 1e9 * u / b, 'o-')
    plt.xticks(np.arange(1, n + 1))
    nbt.ylabelh('$K_d$ (nM)')
    if groupnames is not None:
        plt.legend(h, groupnames, loc='best')

    plt.subplot(4, 1, 4)
    if groupnames is None:
        legendtext_fstates = None
    else:
        legendtext_fstates = \
            ['$F_j$ (%s)' % groupnames[0], '$F_j$ (%s)' % groupnames[1]]
    plot_fstates(fstates, legendtext_fstates)

    figh = plt.gcf()
    plt.show()
    return figh


'''
def plot_fluorescence_from_Lf(solver, experiments,
                              beta, Lfmin, Lfmax,
                              npts=1000, ligandname='L',
                              logx=False, plotgamma=True):
    Lf = np.linspace(Lfmin, Lfmax, npts)
    F = eq.Lf2bsfracs(beta, Lf)[0]
    return
'''


def plot_fdecomp_specdata(solver, experiments, excitation_wavelengths=None,
                          datacolor='k', fitcolor='orange'):
    figs = []
    if excitation_wavelengths is None:
        for i, m in enumerate(solver.models):
            if experiments[i]['type'] != 'spectra':
                continue
            Lf = m.model_state['Lfpowers'][:, 1]
            ii_min = np.argmin(Lf)
            WLmin = m.xvals[np.argmax(m.data[ii_min, :])]
            ii_max = np.argmax(Lf)
            WLmax = m.xvals[np.argmax(m.data[ii_max, :])]
            excitation_wavelengths = np.unique([WLmin, WLmax])
            break
    if excitation_wavelengths is None:
        return figs

    for w in excitation_wavelengths:
        for i, m in enumerate(solver.models):
            if experiments[i]['type'] != 'spectra' or w not in m.xvals:
                continue
            ii = np.flatnonzero(m.xvals == w)[0]

            f, fhat, scalefac = data_and_modelfit(m)
            f = f[:, ii]
            fhat = fhat[:, ii]
            # calculate each binding state's contribution to fluorescence
            f_contribs = m.scale * scalefac * m.regressors \
                * m.coefs[:, ii].reshape(1, -1)
            Lf = m.model_state['Lfpowers'][:, 1]

            nextfig = plt.figure()
            h = plot_for_each_binding_state(Lf, f_contribs, semilogx=True)
            h += plt.semilogx(Lf, fhat, color=fitcolor)
            h += plt.semilogx(Lf, f, color=datacolor)
            for lineobj in h:
                lineobj.set_marker('.')
            plt.title('Fluorescence from spectra'
                      ', $\\lambda_{{ex}}$ = {0} nm'.format(w))

            figs.append(nextfig)
            break  # done with this wavelength
    return figs


def plot_bsfracs_from_Lf(beta, Lfmin, Lfmax, npts=1000,
                         bsnames=None, ligandname='L',
                         logx=False, plotgamma=True):
    # graph ratio of macromolecule in each binding state as a function of free
    # ligand concentration

    assert beta.size > 0, "beta cannot be empty"
    f = plt.figure()

    Lf = np.linspace(Lfmin, Lfmax, npts)
    F = eq.Lf2bsfracs(beta, Lf)[0]

    # plot the binding states as a function of free ligand
    lines = plot_for_each_binding_state(Lf.reshape(-1, 1) * 1e9,
                                        F,
                                        semilogx=logx)

    if plotgamma:
        YL = lines[0].axes.get_ylim()
        gamma = eq.beta2gamma(beta) * 1e9
        n = gamma.size
        # plot over full vertical range
        for i, g in enumerate(gamma):
            plt.plot(g * np.ones(2), YL, '--', color=bscolor(i + 1))
        # plot again such that identical gammas don't overlap completely
        for i, g in enumerate(gamma):
            y = YL[0] + np.array([i / n, (i + 1) / n])
            plt.plot(g * np.ones(2), y, '--', color=bscolor(i + 1))
        plt.xlim([Lfmin * 1e9, Lfmax * 1e9])

    if bsnames is not None:
        plt.legend(lines, bsnames, loc='best')
    plt.xlabel('$[{0}]_{{free}} (nM)$'.format(ligandname))
    plt.ylabel('Binding state fraction')

    return f


def plot_allitc(solver, experiments,
                datacolor='k', fitcolor='orange',
                useLf=False):
    f = plt.figure()
    ylt = ''
    for i, m in enumerate(solver.models):
        if experiments[i]['type'] != 'itc':
            continue
        if ylt is '':
            ylt = m.data_types[0]
        data, datahat, _ = data_and_modelfit(m)
        if useLf:
            Lf = m.model_state['Lfpowers'][:, 1]
            plt.plot(Lf[1:], data, color=datacolor)
            plt.plot(Lf[1:], datahat, color=fitcolor)
        else:
            plt.plot(data, color=datacolor)
            plt.plot(datahat, color=fitcolor)

    if useLf:
        ligandname = experiments[0]['ligand']
        plt.xlabel('$[{0}]_{{free}} (nM)$'.format(ligandname))
    else:
        plt.xlabel('Injection number')
    plt.ylabel(ylt)
    plt.title('ITC experiments')
    return f


def plot_sensitivity_curves(solver, experiments, xi_perturbed, perturbed_x):
    '''
    plot fractional increase in sum of squared errors for each data type and
    overall for each perturbed parameter
    '''
    s = solver.copy()
    npts = perturbed_x.shape[1]  # max points on sensitivity curve
    assert np.mod(npts, 2) == 1, "curves must have an odd number of points"
    etypes = list(np.unique([e['type'] for e in experiments]))  # sorted
    etypeind = np.searchsorted(etypes, [e['type'] for e in experiments])
    j0 = (npts - 1) // 2  # index of unperturbed values at optimum

    f = []
    for i, xi in enumerate(xi_perturbed):
        xvals = perturbed_x[i, :, xi]  # perturbed parameter values
        ssqe_by_etype = np.full((npts, len(etypes)), np.nan)
        ssqe_total = np.full(npts, np.nan)
        for j in range(npts):
            if np.isnan(perturbed_x[i, j, :]).any():
                continue  # out of bounds etc.
            s.loglikelihood(perturbed_x[i, j, :])
            # fixme: take variances into account?
            ssqevec = [((m.data - np.dot(m.regressors, m.coefs * m.scale))**2)
                       .sum() for m in s.models]
            ssqevec = np.array(ssqevec)
            ssqe_total[j] = np.sum(ssqevec)
            for u in range(len(etypes)):
                ssqe_by_etype[j, u] = ssqevec[etypeind == u].sum()
        f.append(plt.figure())
        L = plt.plot(xvals, ssqe_by_etype / ssqe_by_etype[j0])
        L += plt.plot(xvals, ssqe_total / ssqe_total[j0],color='k')
        plt.legend(L, etypes + ['total'], loc='best')
        YL = plt.ylim()
        plt.plot(xvals[[j0, j0]], YL,'k:')
        plt.ylim(YL)
        varname = s.x_names[xi]
        if '\\' in varname[0] or '_' in varname or '{' in varname:
            varname = '$' + varname + '$'
        xlab = plt.xlabel(varname)
        xlab.set_fontsize(np.maximum(xlab.get_fontsize(), 15))
        plt.ylabel('Normalized mean square residual')

    return f

def plot_stoppedflow_onoff(solver, experiments,
                           datacolor='k', fitcolor='orange',
                           logx=False):
    i_off = [i for i, e in enumerate(experiments)
             if e['type'] == 'stoppedflow' and
             e['total ligand'][0] > e['total ligand'][1]]
    i_on = [i for i, e in enumerate(experiments)
            if e['type'] == 'stoppedflow' and
            e['total ligand'][0] <= e['total ligand'][1]]
    both_present = len(i_off) > 0 and len(i_on) > 0

    if logx:
        plotfcn = plt.semilogx
    else:
        plotfcn = plt.plot

    f = []
    f.append(plt.figure())
    if len(i_off) > 0:
        Mtot = np.array([experiments[i]['total macromolecule'].sum()
                         for i in i_off])
        normfac = Mtot / np.min(Mtot)
        for j, i in enumerate(i_off):
            m = solver.models[i]
            e = experiments[i]
            data, datahat, _ = data_and_modelfit(m)
            t = ki.stoppedflow_ftimes(m, e)

            hd = plotfcn(t[1:], data / normfac[j], color=datacolor)
            hm = plotfcn(t[1:], datahat / normfac[j], color=fitcolor)
        plt.legend([hd[0], hm[0]], ['Data', 'Model'], loc='best')
        plt.xlabel('Time')
        plt.ylabel('Fluorescence')
        plt.title('Unbinding')
    if both_present:
        f.append(plt.figure())

    if len(i_on) > 0:
        Mtot = np.array([experiments[i]['total macromolecule'].sum()
                         for i in i_on])
        normfac = Mtot / np.min(Mtot)
        for j, i in enumerate(i_on):
            m = solver.models[i]
            e = experiments[i]
            data, datahat, _ = data_and_modelfit(solver.models[i])
            t = ki.stoppedflow_ftimes(m, e)

            hd = plotfcn(t[1:], data / normfac, color=datacolor)
            hm = plotfcn(t[1:], datahat / normfac, color=fitcolor)
        plt.legend([hd[0], hm[0]], ['Data', 'Model'], loc='best')
        plt.xlabel('Time')
        plt.ylabel('Fluorescence')
        plt.title('Binding')
    return f


def data_and_modelfit(m):
    datahat = np.dot(m.regressors, m.coefs * m.scale)
    data = m.data.copy()
    # if we reweighted the data data before fitting, show the fit in the
    # original units:
    scalefac = 1.0
    if 'normalization_factor' in dir(m):
        scalefac *= m.normalization_factor

    datahat *= scalefac
    data *= scalefac
    return data, datahat, scalefac


def plot_modelfit(m, e=None, index=None,
                  datacolor='k', fitcolor='orange',
                  plot_extras=True,
                  itc_xaxis='ratio',  # ratio, step or Lf
                  itc_permole=True,
                  itc_stepbased=False):
    # assert type(m) == mf.regressionmodel, "m must be a regressionmodel"
    data, datahat, scalefac = data_and_modelfit(m)

    if e is not None and 'data units' in e.keys():
        dataunits = e['data units']
    else:
        dataunits = '???'

    f = plt.figure()

    if e is not None and e['type'] == 'itc':
        if dataunits == '$\\mu$cal' and itc_permole:
            dataunits = 'kcal'
            data *= 1e-9
            datahat *= 1e-9
            scalefac *= 1e-9
        n_inj = data.size  # number of injections
        if itc_xaxis == 'step':
            x = np.arange(1, n_inj)
            xname = "Injection number"
        elif itc_xaxis == 'ratio':
            x = m.model_state['Lt'][1:] / m.model_state['Mt'][1:]
            xname = "${0}_{{total}}:{1}_{{total}}$" \
                    " ratio".format(e['ligand'], e['macromolecule'])
        elif itc_xaxis=='Lf':
            x = m.model_state['Lfpowers'][1:, 1]
            xname = "Free ${0}$".format(e['ligand'])
        else:
            raise ValueError("Invalid itc x-axis: {0}".format(itc_xaxis))

        if e['titration type'] == 'ligand into macromolecule' and itc_permole:
            # calculate moles of injected ligand at each step
            V_injected = np.diff(e['V_other'])
            moles_injectant = e['total_ligand_other_solution'] * V_injected
            data = data / moles_injectant
            datahat = datahat / moles_injectant
            ylt = 'Heat ({0} / mole injectant)'.format(dataunits)
        else:
            assert itc_permole == False, \
                "can''t normalize for reverse titraction"
            ylt = m.data_types[0] + ' ({0})'.format(dataunits)

        hd = plt.plot(x, data, 'o', color=datacolor)
        hm = plt.plot(x, datahat, 's', color=fitcolor)
        if plot_extras:
            YL = hd[0].axes.get_ylim()
            minY = YL[0] - (YL[1] - YL[0])
            maxY = YL[1] + (YL[1] - YL[0])

            dc = m.regressors  # concentration changes
            h = m.coefs.reshape(-1) * m.scale * scalefac
            if itc_stepbased:
                # heat from each binding step, instead of the total heat from
                # each non-ligand-free binding state
                h = np.append(h[0], np.diff(h))
                dc = dc[:, -1::-1].cumsum(axis=1)[:, -1::-1]
            partial_heats = dc * h.reshape(1, -1)
            partial_heats = np.hstack((np.full((n_inj, 1), np.nan),
                                       partial_heats))
            if itc_permole:
                partial_heats = partial_heats / moles_injectant.reshape(-1, 1)
            plot_for_each_binding_state(x,
                                        partial_heats)

            YLnew = hd[0].axes.get_ylim()
            hd[0].axes.set_ylim(np.maximum(YLnew[0], minY),
                                np.minimum(YLnew[1], maxY))
        plt.xlabel(xname)
    elif e is not None and e['type'] == 'spectra':
        hd = plt.plot(m.xvals, data.T, color=datacolor)
        hm = plt.plot(m.xvals, datahat.T, color=fitcolor)
        plt.xlabel('$\\lambda_{ex}$')
        ylt = 'Fluorescence'
    elif e is not None and e['type'] == 'stoppedflow':
        t = ki.stoppedflow_ftimes(m, e)
        hd = plt.plot(t[1:], data, color=datacolor)
        hm = plt.plot(t[1:], datahat, color=fitcolor)
        if plot_extras:
            LjM = m.model_state['LjM']
            if 'regressor_groups' in dir(m):
                LjM = util.collapse_columns(LjM, m.regressor_groups)
            partial_f = LjM * m.coefs.reshape(1, -1) * m.scale * scalefac
            plot_for_each_binding_state(t, partial_f)
        plt.xlabel('Time')
        ylt = 'Fluorescence'
    elif m.datadim == 1:  # generic 1D
        hd = plt.plot(data, color=datacolor)
        hm = plt.plot(datahat, color=fitcolor)
        ylt = m.data_types[0]
    else:  # generic
        hd = plt.plot(data.T, color=datacolor)
        hm = plt.plot(datahat.T, color=fitcolor)
        ylt = None

    if ylt is not None:
        if e['type'] != 'itc':
            ylt += ' ({0})'.format(dataunits)
        plt.ylabel(ylt)
    plt.legend([hd[0], hm[0]], ['Data', 'Model'], loc='best')

    plt.title(modelfit_title(m, e, index=index))
    hd[0].axes.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    return f


def modelfit_title(m, e, index=None):
    title = ''
    if e is not None and 'type' in e.keys():
        if len(title) > 0:
            title += ' '
        title += e['type']
        if e['type'] == 'spectra' and 'buffer' in e.keys():
            title += ' (buffer: {0})'.format(e['buffer'])
        if e['type'] == 'itc' and 'titration type' in e.keys():
            L = e['total_ligand_other_solution'] * 1e6
            M = e['macromolecule_concentration'] * 1e6
            ll = e['ligand']
            mm = e['macromolecule']
            if e['titration type'] == 'ligand into macromolecule':
                ts = '({0} $\\mu$M {1} into {2} $\\mu$M {3})'
                title += ts.format(L, ll, M, mm)
            else:
                ts = '({0} $\\mu$M {1} into {2} $\\mu$M {3})'
                title += ts.format(M, mm, L, ll)
    if e is not None and 'filename' in e.keys():
        if len(title) > 0:
            title += ' '
        title += os.path.basename(e['filename'])
    if e is not None and 'protein_purification' in e.keys():
        if len(title) > 0:
            title += ' '
        title += os.path.basename(e['protein_purification'])
    if index is not None:
        title = '{0}. {1}'.format(index, title)

    return title


def plot_stoppedflow_states(m, experiment):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    t = ki.stoppedflow_ftimes(m, experiment)
    n = m.model_state['LjM'].shape[1]
    cu = 1.0
    if 'concentration_unit' in dir(m):
        cu = m.concentration_unit / 1e-6
    lineobj = []

    for i in range(n):
        lineobj.append(ax1.plot(t, m.model_state['LjM'][:, i] * cu,
                                color=bscolor(i)))
    lineobj.append(ax1.plot(t, m.model_state['Lf'] * cu, color=bscolor(n)))
    lineobj.append(ax2.plot(t, m.model_state['B'] * cu, color=bscolor(n + 1)))
    lineobj.append(ax2.plot(t, m.model_state['LB'] * cu, color=bscolor(n + 2)))

    legendtext = [bsname(i) for i in range(n)] + ['$[L]$', '$[B]$', '$[LB]$']
    lineobj = [v[0] for v in lineobj]
    plt.legend(tuple(lineobj), legendtext, loc='best')
    plt.axes(ax1)
    plt.ylabel('Everything else')
    plt.axes(ax2)
    plt.ylabel('Buffers')

    return fig


def plot_fracsandca(m, experiment, xvals=None):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    Lf = m.model_state['Lfpowers'][:, 1]
    r = m.model_state['bsfracs']
    if r.ndim > 2:
        r = r[0, :, :]  # macromolecule only

    for i in range(r.shape[1]):
        ax1.plot(r[:, i], '.-', color=bscolor(i))
        ax2.semilogy(Lf, 'k.-')
    return fig


def plot_info(solver, experiments):
    fig, ax = plt.subplots()

    etypes = list(set([e['type'] for e in experiments]))
    n = [np.sum([e['type'] == et for e in experiments]) for et in etypes]
    infostr = ', '.join(['{nn} {tt}'.format(nn=n[i], tt=et)
                         for i, et in enumerate(etypes)])

    infostr += '\n' + solver.param_description()
    plt.text(0, 0, infostr, fontdict={'family': 'monospace'})
    ax.set_axis_off()
    plt.show()


    return fig


def plot_globalfit(solver, experiments,
                   savefigs=True,
                   show_fracsandca=True,
                   showspecdecomp=True,
                   filename='global fit.pdf'):

    cu = solver.models[0].concentration_unit
    assert np.all([m.concentration_unit == cu for m in solver.models]), \
        "all models in the solver must have the same concentration unit"
    beta_cu = ki.getbeta_mm_fromsolver(solver)
    # mm is short for macromolecule
    gamma_mm = eq.beta2gamma(beta_cu) * cu
    beta_mm = eq.gamma2beta(gamma_mm)
    nbindingsteps = beta_mm.size
    bsnames = names.bindingstatenames(experiments[0], nbindingsteps)
    ligandname = experiments[0]['ligand']

    figs = []

    figs.append(plot_info(solver, experiments))

    for i, m in enumerate(solver.models):
        figs.append(plot_modelfit(m, experiments[i], i))
        if experiments[i]['type'] in ['spectra', 'itc'] and show_fracsandca:
            figs.append(plot_fracsandca(m, experiments[i]))

    if showspecdecomp:
        figs += plot_fdecomp_specdata(solver, experiments)

    Lfmin = gamma_mm[0] / 10.0
    Lfmax = gamma_mm[-1] * 10.0
    figs.append(plot_bsfracs_from_Lf(beta_mm, Lfmin, Lfmax, 1000,
                                     bsnames=bsnames,
                                     ligandname=ligandname,
                                     logx=True))

    if np.sum([e['type'] == 'stoppedflow' for e in experiments]) > 1:
        figs += plot_stoppedflow_onoff(solver, experiments)
        figs += plot_stoppedflow_onoff(solver, experiments, logx=True)

    if np.sum([e['type'] == 'itc' for e in experiments]) > 1:
        figs.append(plot_allitc(solver, experiments))

    plt.show()
    if savefigs:
        nbt.savefigs_pdf(figs, filename)
