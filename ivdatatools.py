import numpy as np
import os
import csv
import datetime
import copy

version = 0.1


def autodatadir():
    datalocs = [
        '/home/lab/Dropbox (NIG)/in vitro data/',
        '/home/greenberg/Documents/in vitro data/',
        '/data/analysis_dg/in vitro data/',
        '/home/monsees/PhD/data/data_from_silvi/',
        'C:\\Users\\labor01\\Dropbox (NIG)\\in vitro data',
        'D:\db\Dropbox (NIG)\in vitro data'
    ]

    dirok = [os.path.isdir(d) for d in datalocs]
    assert np.any(dirok), "Can\'t find data"
    return datalocs[dirok.index(True)]


def forbidden_args(exptype):
    fa = ['version', 'filename', 'basename', 'relpath', 'importtime', 'type']
    if exptype == 'spectra':
        fa += ['data']  # we'll read it from the file
    return fa


def required_args(exptype):
    '''required input arguments for each experiment type'''
    ra = ['temperature', 'ligand', 'macromolecule', 'pH']
    if exptype == 'spectra':
        pass
    elif exptype == 'itc':
        ra += ['V_M', 'V_other', 'titration type', 'data',
               'total_ligand_other_solution', 'macromolecule_concentration']
    elif exptype == 'stoppedflow':
        ra += ['volumes', 'total ligand', 'total macromolecule']
    return ra


def check_experiment_args(experiment_type, experiment):
    for k in forbidden_args(experiment_type):
        assert k not in experiment.keys(), "forbidden argument: {0}".format(k)

    for k in required_args(experiment_type):
        assert k in experiment.keys(), "missing argument: {0}".format(k)

def import_ivdata(filename, experiment_type,
                  datadir=None,
                  always_solve_eq=False,  # require total ligand/mm/buffer etc.
                  **kwargs):

    experiment = copy.deepcopy(kwargs)  # copy all items in dictionary
    check_experiment_args(experiment_type, experiment)

    if experiment_type == 'spectra':
        '''
        list of keys needed for solving the equilibrium. this means we don't
        take free ligand concentration as a given but rather calculate it from
        total ligand, macromolecule and optionally buffer, and their k_d's
        '''
        keys_for_eqsolve = ['total ligand', 'total macromolecule']
        solveeq = always_solve_eq or 'Lf' not in experiment.keys()

        if solveeq:
            for k in keys_for_eqsolve:
                assert k in experiment.keys(), "missing value: {0}".format(k)
            TL = np.atleast_1d(experiment['total ligand'])
            TM = np.atleast_1d(experiment['total macromolecule'])
            nsteps = np.maximum(TL.size, TM.size)
            if experiment['buffer'] != 'none':
                assert 'total buffer' in experiment.keys(), \
                    "missing value: total buffer"
                TB = np.atleast_1d(experiment['total buffer'])
                nsteps = np.maximum(nsteps, TB.size)
        else:
            assert 'Lf' in experiment.keys(), "missing value: Lf"
            nsteps = experiment['Lf'].size

        assert nsteps > 0, "ligand concentration array is empty"
        if experiment['spectra type'] == 'fluorescence':
            d, w, column_names = importfspeccsv(filename)
            bgname = 'buffer'
        elif experiment['spectra type'] == 'absorption':
            d, w, column_names = importaspeccsv(filename)
            bgname = 'B'
        else:
            raise ValueError(
                "unknown spectra type: {0}".format(experiment_type))

        if d.shape[1] == nsteps + 1:
            experiment['background present'] = True
            bgindex = np.flatnonzero(
                np.array([bgname in cn for cn in column_names]))
            assert bgindex.size > 0, "could not identify BG spectrum"
            assert bgindex.size == 1, "multiple background spectra"
            experiment['background spectrum'] = d[:, bgindex]
            d = np.delete(d, bgindex, axis=1)
        else:
            assert d.shape[1] == nsteps, "wrong number of steps"
            experiment['background present'] = False

        experiment['data'] = d # / experiment['integration time (ms)']
        if experiment['spectra type'] == 'fluorescence':
            experiment['excitation wavelengths'] = w
        elif experiment['spectra type'] == 'absorption':
            experiment['wavelengths'] = w
        else:
            raise ValueError(
                "unknown spectra type: {0}".format(experiment_type))

    elif experiment_type == 'itc':
        '''
        for ITC data, right now most of the work is done before calling this
        function. we still get time stamps, do field checking, etc. though.
        '''

    elif experiment_type == 'stoppedflow':
        if 'buffer' in experiment['buffer'] != 'none':
            assert 'total buffer' in experiment.keys(), \
                "total buffer concentration must be available when buffer is" \
                "not none"
        data = ParseProDataCSV(filename)
        assert data.shape[1] > 1, "at least 2 columns of data are required"
        experiment['time'] = data[:, 0]
        experiment['data'] = data[:, 1:]

    else:
        raise ValueError("unknown experiment type: {0}".format(experiment_type))

    experiment['type'] = experiment_type
    experiment['version'] = version
    experiment['filename'] = filename
    experiment['basename'] = os.path.basename(filename)
    if datadir is not None:
        experiment['relpath'] = os.path.relpath(filename, datadir)
    else:
        experiment['relpath'] = None
    experiment['importtime'] = datetime.datetime.now(
    ).strftime('%Y-%m-%d %H:%M:%S')

    return experiment


def getsfdata(datadir,
              sample_index=1,
              experiment_type='on',
              temperature=37, datasetid=1):
    '''
    retrieve stopped flow data
    '''
    if experiment_type == 'on':
        exptypedir = 'on rates - BAPTA buffer system'
    elif experiment_type == 'off':
        exptypedir = 'off rates'
    else:
        raise ValueError("unknown experiment type: %s" % experiment_type)

    subdir = os.path.join(datadir, 'Stopped-Flow',
                          exptypedir, str(temperature) + 'C')

    assert os.path.isdir(subdir), "directory %s does not exist" % subdir

    if os.path.isdir(os.path.join(subdir, 'CSV')):

        assert datasetid == 1, "only 1 dataset, dataset id must be 1 or absent"

    else:

        datasetdirs = [d for d in os.listdir(subdir)
                       if d.startswith('set ' + str(datasetid)) and
                       os.path.isdir(os.path.join(subdir, d))
                      ]
        assert len(datasetdirs) > 0, "dataset with id %d not found" % datasetid
        assert len(datasetdirs) <= 1, "duplicate dataset w/ id %d" % datasetid
        subdir = os.path.join(subdir, datasetdirs[0])

    subdir = os.path.join(subdir, 'CSV')
    assert os.path.isdir(subdir), "directory %s does not exist" % subdir

    csvfiles = [fi for fi in os.listdir(subdir)
                if fi.lower().endswith('.csv') and
                (fi.startswith('all traces sample ' + str(sample_index)) or
                 fi.startswith(str(sample_index) + '_all traces')
                ) and 'average' not in fi]
    assert len(csvfiles) > 0, "sample %d not found" % sample_index
    assert len(csvfiles) <= 1, "duplicate sample with id %d" % sample_index

    filename = os.path.join(subdir, csvfiles[0])

    data = ParseProDataCSV(filename)
    assert data.shape[1] > 1, "at least 2 columns of data are required"
    tobs = data[:, 0]
    f = data[:, 1:]

    return (tobs, f)


def importaspeccsv(filename):
    with open(filename, newline='') as csvfile:
        r = csv.reader(csvfile, delimiter=',')

        rowdata = []
        for row in r:
            rowdata.append(row)

    ncolumns = len(rowdata[4])
    dataind = 7 + [q[0] for q in rowdata].index('Data:')

    nwavelengths = 0
    for i in range(dataind, len(rowdata)):
        if len(rowdata[i]) != ncolumns:
            break
        if np.any([len(c) == 0 for c in rowdata[i]]):
            break  # fixme could fill in partial data with nan's?
        nwavelengths += 1

    data = np.full((nwavelengths, ncolumns - 1), np.nan)
    wavelengths = np.full(nwavelengths, np.nan)

    for i in range(0, nwavelengths):
        data[i, :] = rowdata[dataind + i][1:]
        wavelengths[i] = rowdata[dataind + i][0]

    column_names = rowdata[4][1:]

    return (data, wavelengths, column_names)


def importfspeccsv(filename):
    with open(filename, newline='') as csvfile:
        r = csv.reader(csvfile, delimiter=',')

        rowdata = []
        for row in r:
            rowdata.append(row)

    ncolumns = len(rowdata[4])
    nwavelengths = 0
    for i in range(4, len(rowdata)):
        if len(rowdata[i]) != ncolumns:
            break
        if np.any([len(c) == 0 for c in rowdata[i]]):
            break  # fixme could fill in partial data with nan's?
        nwavelengths += 1

    assert ncolumns % 2 == 0, "even number of columns required"
    data = np.full((nwavelengths, ncolumns // 2), np.nan)
    wavelengths = np.full(nwavelengths, np.nan)

    for i in range(0, nwavelengths):
        data[i, :] = rowdata[4 + i][1::2]
        wavelengths[i] = rowdata[4 + i][0]

    column_names = rowdata[2][0::2]

    return (data, wavelengths, column_names)


def ParseProDataCSV(filename):
    with open(filename, newline='') as csvfile:
        r = csv.reader(csvfile, delimiter=',')

        # pass through the csv file to find where the data start and stop
        startrow, endrow, ncolumns = None, None, None
        n = 0
        for row in r:
            if startrow is None:  # check for the start of the data
                if len(row) > 0 and type(row[0]) == str:
                    issf = len(row) == 2 and type(row[1]) == str and \
                        row[0] == 'Time' and row[1] == 'Repeat'
                    isspec = len(row) == 2 and type(row[1]) == str and \
                        row[0] == 'Wavelength' and row[1] == ''
                    isweird = len(row) == 2 and type(row[1]) == str and \
                        row[0] == 'Time' and row[1] == 'Wavelength'
                    if issf or isspec or isweird:
                        startrow = n + 2

            elif n > startrow:  # find the end
                if len(row) == 0:
                    endrow = n
                    break
                else:
                    # "excel row indexing"
                    assert ncolumns == len(row), \
                        "error parsing %s:\nexpected %d columns on line %d," \
                        "found %d" % (filename, ncolumns, n + 1, len(row))

            elif n == startrow:
                ncolumns = len(row)
                if ncolumns == 0:
                    endrow = n
                    break
            n += 1

        # check for success
        if startrow is None:
            raise EOFError("No data found in file: %s" % filename)
        if endrow is None:
            raise EOFError("File not terminated properly: %s" % filename)
        if endrow <= startrow:
            raise IOError("No data in file: %s" % filename)

        csvfile.seek(0)  # rewind file

        data = np.full((endrow - startrow, ncolumns), np.nan)
        n = 0
        for row in r:
            if n >= startrow:
                if n >= endrow:
                    break
                data[n - startrow, :] = np.array(row, dtype=float)
            n += 1

        return data


def read_itc_configtxt(configPath):
    raw_config = []
    with open(configPath, "r") as configFile:
        for line in configFile:
            row = str.split(str(line), "\t")
            while '' in row:
                row.remove('')
            row[np.size(row) - 1] = row[np.size(row) - 1].split("\n")[0]
            raw_config.append(row)
    return raw_config


def read_nitpic_datfile(path):
    data = np.array([])
    with open(path, "r") as datfile:
        next(datfile)
        for line in datfile:
            heatvalstr = line.split('\t')[0]
            if heatvalstr != '':
                data = np.append(data, float(heatvalstr))
    return data[1:]  # always remove 0th titration step


def filter_experiments(experiments,
                       missing_ok=True,
                       Ttol=0.5,
                       **kwargs):
    '''return experiments matching a list of criteria'''
    elist = []
    for e in experiments:
        ok = True
        for key in kwargs.keys():
            if key not in e.keys():
                if missing_ok:
                    continue
                ok = False
                break

            v = kwargs[key]
            ev = e[key]

            if key == 'temperature':
                if np.abs(v - ev) > Ttol:
                    ok = False
                    break
                else:
                    continue

            if not isinstance(v, list):
                v = [v]

            if e[key] not in v:
                ok = False
                break

        if ok:
            elist.append(copy.deepcopy(e))

    return elist
