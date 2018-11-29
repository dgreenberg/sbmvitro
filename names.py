import numpy as np


def betanames(nbindingsteps):  # \beta_0 = 1 by definition
    return ['\\beta_{{{0}}}'.format(j) for j in range(1, nbindingsteps + 1)]


def rateconstantnames(x):
    if type(x) is int:
        kon_names = ['k_{{+{0}}}'.format(i + 1) for i in range(x)]
        koff_names = ['k_{{-{0}}}'.format(i + 1) for i in range(x)]
    elif type(x) is list:
        for n in x:
            assert type(n) is str, "x must be an integer or list of strings"
        kon_names = ['k_{{+{0}}}'.format(i) for i in x]
        koff_names = ['k_{{-{0}}}'.format(i) for i in x]
    else:
        raise TypeError("x must be an integer or list of strings")
    return koff_names, kon_names


def gdgtnames(x):
    # \\gamma[0], \\Delta\\gamma, \\tau
    if type(x) is int:
        gdg_names = ['\\gamma_{1}'] + \
                    ['\\Delta\\gamma_{{{0}}}'.format(i + 1)
                     for i in range(1, x)]
        tau_names = ['\\tau_{{{0}}}'.format(i + 1)
                     for i in range(x)]
    elif type(x) is list:
        for n in x:
            assert type(n) is str, "x must be an integer or list of strings"
        gdg_names = ['\\gamma_{{{0}}}'.format(i) for i in x]
        tau_names = ['\\tau_{{{0}}}'.format(i) for i in x]
    else:
        raise TypeError("x must be an integer or list of strings")
    return gdg_names, tau_names


def bindingstatenames(experiment, nbindingsteps):  # fixme not from exp. dict.
    if 'ligand' in experiment.keys():
        ligandname = experiment['ligand']
    else:
        ligandname = 'L'

    if ligandname[-1] == '+' or ligandname[-1] == '-':
        # e.g. if ligandname is Ca2+, identify the base name as Ca
        ligandbasename = ligandname[0:-1]
        isdigit = np.array([c.isdigit() for c in ligandbasename])
        assert not np.all(
            isdigit), "invalid ligand name: {0}".format(ligandname)
        n = np.flatnonzero(~isdigit).max()
        ligandbasename = ligandbasename[0:n + 1]
    else:
        ligandbasename = ligandname

    if 'macromolecule' in experiment.keys():
        macromolecule = experiment['macromolecule']
    else:
        macromolecule = 'M'

    bsnames = [macromolecule]
    if nbindingsteps > 0:
        bsnames.append(ligandbasename + macromolecule)
    for i in range(2, nbindingsteps + 1):
        if i < 10:
            bsnames.append('{0}_{1}{2}'.format(
                ligandbasename, i, macromolecule))
        else:
            bsnames.append('{0}_{{{1}}}{2}'.format(
                ligandbasename, i, macromolecule))
    return bsnames


def itcregressornames(experiment, nbindingsteps):
    bsnames = bindingstatenames(experiment, nbindingsteps)
    return ['\\Delta' + bsn for bsn in bsnames[1:]]


def isbetaname(name):
    return name.startswith('\\beta_{') and name.endswith('}')


def isgammaname(name):
    return name.startswith('\\gamma_{') and name.endswith('}')


def israteconstantname(name):
    return (name.startswith('k_{+') or name.startswith('k_{-')) and \
        name.endswith('}')


def Ka_names2rateconstantnames(Ka_names):
    koff_names, kon_names = [], []
    for name in Ka_names:
        subname = name[len('Ka_{'):-1]
        koff_names.append('k_{{-{0}}}'.format(subname))
        kon_names.append('k_{{+{0}}}'.format(subname))
    return (koff_names, kon_names)


def binding_param_names(nbindingsteps):
    gdg_names, tau_names = gdgtnames(nbindingsteps)
    koff_names, kon_names = rateconstantnames(nbindingsteps)
    beta_names = betanames(nbindingsteps)
    return gdg_names + tau_names + koff_names + kon_names + beta_names
