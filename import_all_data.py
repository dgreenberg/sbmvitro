'''
import functions to get data from various formats into standardized
lists of dicts
'''
import os
import warnings
import ivdatatools as ivd
import numpy as np
import datetime
import copy

def standardinfo():
    ''' standard experimental conditions '''
    info = {'ligand': 'Ca2+',
            'macromolecule': 'GCaMP6s',
            'pH': 7.2,
            'ionic strength': 0.15}  # M
    return info


def import_stoppedflowdata(datadir):

    experiments = []

    sfdir = os.path.join(datadir, 'Stopped-Flow')

    # off rates
    info = standardinfo()
    info.update({
        'excitation wavelength': 488.0,
        'slit width (nm)': 0.5,
        'emission filter': 'cut-off filter 515 nm',
        'volumes': np.array([1.0, 1.0]),  # 1:1 mixing. fixme: actual volumes?
        'total macromolecule': np.array([1e-6, 0.0]),  # 1 uM protein
        'total buffer': np.atleast_2d([0.0, 10e-3]),  # 10 mM BAPTA
        'buffer': 'bapta',
        'dead time (ms)': 1.5,
        'instrument': 'sx20',
        'pathlength': 10
    })
    # 22 deg.
    d = os.path.join(sfdir, 'off rates', '22C', 'CSV')
    ''' skip this file, wavelength was different for each repeat
    filename = os.path.join(d, '1_all traces 160119.csv')
    info.update({'total ligand': np.array([10e-6, 0.0]),
                 'experiment_date': '2016-01-19',
                 'pmt voltage': 463.425})
    experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                         datadir=datadir,
                                         protein_purification='pu4-2',
                                         temperature=22.17,
                                         **info))
    experiments[-1]['data'] = experiments[-1]['data'][:, 0:1]  # 488 nm only
    '''
    filename = os.path.join(d, '2_all traces 160308 Pu5.csv')
    info.update({'total ligand': np.array([20e-6, 0.0]),
                 'experiment_date': '2016-03-08',
                 'pmt voltage': 399.994})
    experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                         datadir=datadir,
                                         protein_purification='pu5',
                                         refolded=True,
                                         temperature=22.07,
                                         **info))
    ''' skip this file, wavelength was different for each repeat
    filename = os.path.join(d, '3_all traces 160308 Pu6.csv')
    info.update({'total ligand': np.array([20e-6, 0.0]),
                 'experiment_date': '2016-03-08',
                 'pmt voltage': 399.994})
    experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                         datadir=datadir,
                                         protein_purification='pu6',
                                         refolded=True,
                                         temperature=22.02,
                                         **info))
    experiments[-1]['data'] = experiments[-1]['data'][:, 0:1]  # 488 nm only
    '''
    # 37 deg.
    d = os.path.join(sfdir, 'off rates', '37C', 'CSV')

    filename = os.path.join(d, '1_all traces 160129.csv')
    info.update({'total ligand': np.array([10e-6, 0.0]),
                 'experiment_date': '2016-01-29',
                 'pmt voltage': 430.603})
    experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                         datadir=datadir,
                                         protein_purification='pu5',
                                         refolded=True,
                                         temperature=37.01,
                                         **info))
    experiments[-1]['data'] = experiments[-1]['data'][:, 0:1]  # 488 nm only
    ''' skip this file, wavelength was different for each repeat, and same as 1.
    filename = os.path.join(d, '2_all traces 160129.csv')
    info.update({'total ligand': np.array([10e-6, 0.0]),
                 'experiment_date': '2016-01-29',
                 'pmt voltage': 430.801})
    experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                         datadir=datadir,
                                         protein_purification='pu5',
                                         refolded=True,
                                         temperature=37.26,
                                         **info))
    experiments[-1]['data'] = experiments[-1]['data'][:, 0:1]  # 488 nm only
    '''
    filename = os.path.join(d, '3_all traces 160308.csv')
    info.update({'total ligand': np.array([20e-6, 0.0]),
                 'experiment_date': '2016-03-08',
                 'pmt voltage': 399.994})
    experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                         datadir=datadir,
                                         protein_purification='pu6',
                                         refolded=True,
                                         temperature=36.96,
                                         **info))

    # on rates, BAPTA buffered
    info['total buffer'] = np.atleast_2d([50e-6, 2e-3])
    info['experiment_date'] = '2016-02-26'
    Lt2 = 1e-3 * np.array([0.0, 0.6, 1.0, 1.4, 1.6, 1.8])

    for i in range(1, 7):
        info['total ligand'] = np.array([0.0, Lt2[i - 1]])
        info['pmt voltage'] = 399.994

        # 37 deg.
        info['temperature'] = 36.96  # slight variation but who cares
        d = os.path.join(sfdir, 'on rates - BAPTA buffer system', '37C')
        filename = os.path.join(d, 'set 1 - 160303', 'CSV',
                                'all traces sample {0}.csv'.format(i))
        experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                             datadir=datadir,
                                             buffer_batch='3',
                                             protein_purification='pu6',
                                             refolded=True,
                                             **info))
        if i != 2:  # set 2 sample 2 shows a downward trend over shots
            filename = os.path.join(d, 'set 2 - 160308', 'CSV',
                                    'all traces sample {0}.csv'.format(i))
            experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                                 datadir=datadir,
                                                 buffer_batch='4',
                                                 protein_purification='pu5',
                                                 refolded=True,
                                                 **info))
        filename = os.path.join(d, 'set 3 - 160308', 'CSV',
                                'all traces sample {0}.csv'.format(i))
        experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                             datadir=datadir,
                                             protein_purification='pu6',
                                             buffer_batch='4',
                                             refolded=True,
                                             **info))
        # 22 deg.
        info['temperature'] = 22.07  # slight variation but who cares
        d = os.path.join(sfdir, 'on rates - BAPTA buffer system', '22C')
        if i not in [1]:
            filename = os.path.join(d, 'set 1 - 160303', 'CSV',
                                    'all traces sample {0}.csv'.format(i))
            experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                                 datadir=datadir,
                                                 buffer_batch='3',
                                                 protein_purification='pu6',
                                                 refolded=True,
                                                 **info))
        filename = os.path.join(d, 'set 2 - 160304', 'CSV',
                                'all traces sample {0}.csv'.format(i))
        experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                             datadir=datadir,
                                             buffer_batch='4',
                                             protein_purification='pu5',
                                             refolded=True,
                                             **info))
        filename = os.path.join(d, 'set 3 - 160304', 'CSV',
                                'all traces sample {0}.csv'.format(i))
        experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                             datadir=datadir,
                                             buffer_batch='4',
                                             protein_purification='pu6',
                                             refolded=True,
                                             **info))
        if i not in [1, 3, 5]:
            continue  # missing files
        filename = os.path.join(d, 'set 4 - 160308 additional BAPTA batch',
                                'CSV',
                                'all traces sample {0}.csv'.format(i))
        experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                             datadir=datadir,
                                             buffer_batch='5',
                                             protein_purification='pu6',
                                             refolded=True,
                                             **info))

    # on rates, (mostly)  unbuffered
    d = os.path.join(sfdir, 'GCaMP6s on-rates free')
    # skipping 160219 folder, due to likely presence of EGTA (higher ca->slower)
    info['total macromolecule'] = np.array([2e-6, 0.0])
    info['temperature'] = 22.12
    info['buffer'] = 'none'
    del info['total buffer']
    Lt = 1e-6 * np.array([0, 2, 4, 10, 20, 0, 1200, 2000], dtype=float)

    for i in range(1, 9):
        filedir = os.path.join(d, '160226 GCaMP6s on rate',
                               'sample {0}'.format(i))
        info['total ligand'] = np.array([0.0, Lt[i - 1]])
        info['experiment_date'] = '2016-01-29'
        for j in [1]:  # no need for multiple repeates for now, need to merge
            filename = os.path.join(filedir,
                                    'Kin sample {0} 0000{1}.csv'.format(i, j))
            if i == 1:
                info['pmt voltage'] = 399.994
            else:
                info['pmt voltage'] = 360.001
            if i <= 5:
                info['total buffer'] = np.zeros((1, 2))
            elif i == 6:
                info['total buffer'] = np.atleast_2d(np.append(0.0, 10e-3))
            else:
                info['total buffer'] = np.atleast_2d(np.append(0.0, 2e-3))
            experiments.append(ivd.import_ivdata(filename, 'stoppedflow',
                                                 datadir=datadir,
                                                 protein_purification='pu5',
                                                 refolded=True,
                                                 **info))

    # on rates, buffered, new batch
    info = standardinfo()
    info.update({
        'excitation wavelength': 488.0,
        'volumes': np.array([1.0, 1.0]),  # 1:1 mixing. fixme: actual volumes?
        'buffer': 'bapta',
        'dead time (ms)': 1.5,
        'instrument': 'sx20',
    })

    d = os.path.join(sfdir, '11_2017_37C', '171122 on rates')
    filename = os.path.join(d, '171122_GC6s_BAPTA_on_37degC.csv')

    #temporary hack, csv file in wrong format:
    #data = ivd.ParseProDataCSV(filename)
    with open(filename, newline='') as csvfile:
        r = ivd.csv.reader(csvfile, delimiter=',')

        rowdata = []
        for row in r:
            rowdata.append(row)
    data = np.full((len(rowdata) - 1, 13), np.nan)
    for i in range(1, len(rowdata)):
        data[i - 1, 0] = rowdata[i][0]
        data[i - 1, 1:] = rowdata[i][1::2]
    assert data.shape[1] == 13, "incorrect data size"

    Lt1 = 0.0  # no calcium added to the GCaMP6s cell
    Lt2 = np.linspace(0.4, 4.8, 12) * 1e-3

    timestr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for i in range(0, Lt2.size):
        exp = copy.deepcopy(info)  # copy all items in dictionary
        exp.update({
            'refolded': False,
            'protein_purification': 'pp_1708',
            'pmt voltage': 380.0,
            'temperature': 37.0,
            'buffer_batch': 'bb_171120',
            'total macromolecule': np.array([1.9992e-6, 0.0]),
            'total buffer': np.atleast_2d([1.9804e-3, 2.0e-3]),
            'total ligand': np.array([Lt1, Lt2[i]]),
            'time': data[:, 0],
            'data': data[:, i + 1:i + 2]
            })
        ivd.check_experiment_args('stoppedflow', exp)
        exp.update({'version': ivd.version, 'filename': filename,
            'basename': os.path.basename(filename),
            'relpath': os.path.relpath(filename, datadir),
            'importtime': timestr, 'type': 'stoppedflow'})
        experiments.append(exp)

    # off rates, buffered, new batch
    d = os.path.join(sfdir, '11_2017_37C', '171110 off rates')
    filename = os.path.join(d, 'GC6s_BAPTA_offrate_37_kin_alltraces.csv')
    data = ivd.ParseProDataCSV(filename)
    assert data.shape[1] == 7, "incorrect data size"
    Lt1 = np.array([5.0, 10.0, 20.0, 30.0, 50.0, 100.0]) * 1e-6
    Lt2 = 0.0

    #last condition only due to possibly stoichiometry errors
    timestr = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    for i in range(5, data.shape[1] - 1):
        exp = copy.deepcopy(info)  # copy all items in dictionary
        exp.update({
            'refolded': False,
            'protein_purification': 'pp_1708',
            'pmt voltage': 350.0,
            'temperature': 37.0,
            'buffer_batch': 'bb_171120',
            'total macromolecule': np.array([2e-6, 0]),
            'total buffer': np.atleast_2d([50e-6, 10.0e-3]),
            'total ligand': np.array([Lt1[i], Lt2]),
            'time': data[:, 0],
            'data': data[:, i + 1:i + 2]
            })
        ivd.check_experiment_args('stoppedflow', exp)
        exp.update({'version': ivd.version, 'filename': filename,
            'basename': os.path.basename(filename),
            'relpath': os.path.relpath(filename, datadir),
            'importtime': timestr, 'type': 'stoppedflow'})
        experiments.append(exp)

    return experiments


def import_specdata(datadir):
    return import_fspecdata(datadir)
    #return import_aspecdata(datadir) + import_fspecdata(datadir)


def import_aspecdata(datadir):
    experiments = []

    specdir = os.path.join(datadir, 'BAPTA absorption')

    info = standardinfo()
    info.update({
        'macromolecule': 'bapta',
        'temperature': 37.0,
        'data units': 'optical density',
        'spectra type': 'absorption',
        'buffer': 'none',
        'buffer_batch': '171120'
    })

    Mt = np.full(10, 250e-6)
    Lt = np.linspace(0, 450e-6, 10)
    info.update({'total ligand': Lt, 'total macromolecule': Mt})
    filename = os.path.join(specdir, '171128_250BxCa_250BxCa.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            experiment_date='2017-11-29',
                            **info)
    exp['total ligand'] = exp['total ligand'][1:]
    # first condition is another BAPTA batch, don't use
    exp['total macromolecule'] = exp['total macromolecule'][1:]
    exp['data'] = exp['data'][:, 1:]
    experiments.append(exp)

    # syringes have two different BAPTA batches, don't use
    '''
    Mt = np.full(13, 250e-6)
    Lt = np.linspace(0, 300e-6, 13)
    info.update({'total ligand': Lt, 'total macromolecule': Mt})
    filename = os.path.join(specdir, '171129_250B_250BxCa.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            experiment_date='2017-11-29',
                            **info)
    exp['total ligand'] = exp['total ligand'][1:]
    exp['total macromolecule'] = exp['total macromolecule'][1:]
    exp['data'] = exp['data'][:, 1:]
    experiments.append(exp)

    Mt = np.full(13, 125e-6)
    Lt = np.linspace(0, 150e-6, 13)
    info.update({'total ligand': Lt, 'total macromolecule': Mt})
    filename = os.path.join(specdir, '171130_125B_125BxCa.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            experiment_date='2017-11-30',
                            **info)
    exp['total ligand'] = exp['total ligand'][1:]
    exp['total macromolecule'] = exp['total macromolecule'][1:]
    exp['data'] = exp['data'][:, 1:]
    experiments.append(exp)
    '''

    #not sure why we're not using this as of now:
    ''' 0-50-100-150-200-250-300-350-400-450-500-550-600
    V_M = np.full(16, 1e-3)  # L

    Vseq1 = np.full(10, 5e-6) # added volume per step, L
    Ltseq1 = 1e-3  # M
    Vseq2 = np.full(5, 2.5) # added volume per step, L
    Ltseq2 = 10e-3  # M

    V = V_M + np.hstack((0.0, Vseq1, Vseq2)).cumsum()  # total volume in L
    Lt = np.hstack((0.0, Vseq1 * Ltseq1, Vseq2 * Ltseq2)).cumsum() / V

    Mt = 25e-6 * V[0] / V
    info.update({
        'total macromolecule': Mt
    })
    filename = os.path.join(specdir, 'BAPTA_2017_08_07_10_45_28_25µM_BAPTA.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            experiment_date='2017-08-07',
                            **info)
    experiments.append(exp)

    Mt = 50e-6 * V[0] / V
    info.update({
        'total macromolecule': Mt
    })
    filename = os.path.join(specdir, 'BAPTA_2017_08_07_10_29_47_50µM_BAPTA.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            experiment_date='2017-08-07',
                            **info)
    experiments.append(exp)

    Mt = 75e-6 * V[0] / V
    info.update({
        'total macromolecule': Mt
    })
    filename = os.path.join(specdir, 'BAPTA_2017_08_07_11_15_37_75µM_BAPTA.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            experiment_date='2017-08-07',
                            **info)
    experiments.append(exp)
    '''

    return experiments


def import_fspecdata(datadir):
    experiments = []

    specdir = os.path.join(datadir, 'Fluorescence spectroscopy - eq binding',
                           'GCaMP6s equilibrium binding - final data set')

    # invitrogen data
    d = os.path.join(specdir, 'equilibrium binding GCaMP6s - Invitrogen - 22C')
    # free calcium from Invitrogen data sheets etc.:
    Lf = np.array([0, 1.859E-08, 4.183E-08, 7.171E-08, 1.115E-07,
                   1.673E-07, 2.509E-07, 3.904E-07, 6.692E-07, 0.0000015,
                   0.000039])

    info = standardinfo()
    info.update({
        'temperature': 22.0,
        'buffer': 'egta',
        'refolded': True,
        'Lf': Lf,
        'emission wavelength': 540.0,
        'lamp power (W)': 70.,
        'instrument': 'pti',
        'maximum data value': 4e6,
        'integration time (ms)': 1000.,
        'data units': 'photons / second',
        'spectra type': 'fluorescence'
    })

    filename = os.path.join(d, 'GCaMP_spectra_Invitrogen_151015.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            protein_purification='pu1',
                            experiment_date='2015-10-15',
                            **info)
    experiments.append(exp)

    filename = os.path.join(d, 'GCaMP_spectra_Invitrogen_151123.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            protein_purification='pu2',
                            experiment_date='2015-11-23',
                            **info)
    experiments.append(exp)

    # bapta data
    d = os.path.join(specdir, 'equilibrium binding GCaMP6s - BAPTA - 22C 37C')

    info = standardinfo()
    info.update({
        'buffer': 'bapta',
        'total macromolecule': 2e-6,
        'total buffer': 2e-3,
        'total ligand': np.linspace(0.0, 2.0, 11) * 1e-3,
        'emission wavelength': 540.0,
        'pH': 7.2,
        'integration time (ms)': 200.,
        'instrument': 'pti',
        'maximum data value': 4e6,
        'neutral density filter (OD)': 0.0,
        'lamp power (W)': 70.,
        'slit width (nm)': 0.5,
        'spectra type': 'fluorescence'
    })

    # for the next 2 files xls says 24th? fixme
    filename = os.path.join(d, 'spectra 160223 GCaMP bapta batch2 22C.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            buffer_batch='3',
                            protein_purification='pu5',
                            refolded=True,
                            temperature=22.0,
                            n_averaged=2,
                            experiment_date='2016-02-23',
                            **info)
    del exp['background spectrum']  # corrupted by GCaMP6s signal
    exp['background present'] = False
    experiments.append(exp)

    filename = os.path.join(d, 'spectra 160223 GCaMP bapta batch2 37C.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            buffer_batch='3',
                            protein_purification='pu5',
                            refolded=True,
                            temperature=37.0,
                            experiment_date='2016-02-23',
                            **info)
    del exp['background spectrum']  # corrupted by GCaMP6s signal
    exp['background present'] = False
    experiments.append(exp)

    info['integration time (ms)'] = 500.
    info['slit width (nm)'] = 0.3

    filename = os.path.join(d, 'spectra 160225 GCaMP bapta batch2 22C.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            buffer_batch='3',
                            protein_purification='pu5',
                            refolded=False,
                            temperature=22.0,
                            experiment_date='2016-02-25',
                            **info)
    del exp['background spectrum']  # corrupted by GCaMP6s signal
    exp['background present'] = False
    experiments.append(exp)

    #missing spectra: bapta batch 4, 16.03.10, Pu6, not refolded

    filename = os.path.join(d, 'spectra 160225 GCaMP bapta batch2 37C.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            buffer_batch='3',
                            protein_purification='pu5',
                            refolded=False,
                            temperature=37.0,
                            experiment_date='2016-02-25',
                            **info)
    del exp['background spectrum']  # corrupted by GCaMP6s signal
    exp['background present'] = False
    experiments.append(exp)

    # unbuffered spectra to mimic ITC experiments
    # why are there 3 sets with 3 different protein concentrations? fixme
    # were they all taken on the same day? fixme
    # is pu9 refolded? fixme
    d = os.path.join(specdir, 'ITC-spectra combi')

    # volume is in liters
    V_M = np.full(20, 407.0) * 1e-6  # L
    V_other = np.hstack((0.0, 0.4, 0.4 + np.arange(1, 19) * 4.0)) * 1e-6  # L
    Lt = 400e-6 * V_other / (V_M + V_other)

    info = standardinfo()
    info.update({
        'temperature': 22.0,
        'buffer': 'none',
        'emission wavelength': 540.0,
        'protein_purification': 'pu9',
        'lamp power (W)': 70,
        'slit width (nm)': 1.5,
        'neutral density filter (OD)': 2.0,
        'integration time (ms)': 100,
        'n_averaged': 2,
        'total ligand': Lt,
        'V_M': V_M,  # volume of macromolecule-containing solution
        'V_other': V_other,  # volume of other solution
        'L_other': 400e-6,  # M
        'L0': 100e-9,  # M. total concentration of L in M solution. Guess.
        'instrument': 'pti',
        'maximum data value': 4e6,
        'spectra type': 'fluorescence'
    })

    M0 = 19.6e-6  # from A280
    Mt = M0 * V_M / (V_M + V_other)
    info.update({'total macromolecule': Mt})
    filename = os.path.join(d, 'GCaMP6s_fluorescence spectra set1.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            experiment_date='2016-05-27',
                            **info)
    experiments.append(exp)

    M0 = 17.0e-6  # from A280
    Mt = M0 * V_M / (V_M + V_other)
    info.update({'total macromolecule': Mt})
    filename = os.path.join(d, 'GCaMP6s_fluorescence spectra set2.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            experiment_date='2016-05-27',
                            **info)
    experiments.append(exp)

    M0 = 20.0e-6  # from A280
    Mt = M0 * V_M / (V_M + V_other)
    info.update({'total macromolecule': Mt})
    filename = os.path.join(d, 'GCaMP6s_fluorescence spectra set3.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            datadir=datadir,
                            experiment_date='2016-05-27',
                            **info)
    experiments.append(exp)

    # itc / spectra combi at 37 with buffered and unbuffered spectra
    d = os.path.join(specdir, 'ITC-spectra combi',
                     '37C - including buffered spectra from same protein batch')
    V_M = np.full(20, 407.0) * 1e-6  # L
    V_other = np.hstack((0.0, 0.4, 0.4 + np.arange(1, 19) * 4.0)) * 1e-6  # L
    Lt = 500e-6 * V_other / (V_M + V_other)

    info = standardinfo()
    info.update({
        'temperature': 37.0,
        'buffer': 'none',
        'emission wavelength': 540.0,
        'protein_purification': 'pu12',
        'lamp power (W)': 70,
        'slit width (nm)': 1.0,
        'neutral density filter (OD)': 2.0,
        'integration time (ms)': 100,
        'n_averaged': 2,
        'total ligand': Lt,
        'V_M': V_M,  # volume of macromolecule-containing solution
        'V_other': V_other,  # volume of other solution
        'L_other': 400e-6,  # M
        'L0': 100e-9,  # M. total concentration of L in M solution. Guess.
        'instrument': 'pti',
        'maximum data value': 4e6,
        'spectra type': 'fluorescence'
    })

    # unbuffered spectra. not using 21.07.16 since rmax is way too low
    M0 = 20.0e-6  # from A280
    Mt = M0 * V_M / (V_M + V_other)
    info.update({'total macromolecule': Mt})
    filename = os.path.join(d, '160722 ITC experiment in PTI 37C set1.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            refolded=True,
                            datadir=datadir,
                            experiment_date='2016-07-22',
                            **info)
    experiments.append(exp)
    M0 = 22.0e-6  # from A280
    Mt = M0 * V_M / (V_M + V_other)
    info.update({'total macromolecule': Mt})
    filename = os.path.join(d, '160722 ITC experiment in PTI 37C set2.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            refolded=True,
                            datadir=datadir,
                            experiment_date='2016-07-22',
                            **info)
    experiments.append(exp)
    filename = os.path.join(d, '160722 ITC experiment in PTI 37C set3.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            refolded=True,
                            datadir=datadir,
                            experiment_date='2016-07-22',
                            **info)
    experiments.append(exp)

    d = os.path.join(d, 'buffered spectra')
    # buffered spectra
    V_M = np.full(11, 400.0) * 1e-6  # L
    V_other = np.arange(11) * 4e-6  # L
    M0 = 2e-6
    Mt = M0 * V_M / (V_M + V_other)
    Lt = 20e-3 * V_other / (V_M + V_other)
    Bt = 2e-3 * V_M / (V_M + V_other)
    info.update({'total macromolecule': Mt,
                 'slit width (nm)': 0.6,
                 'buffer': 'bapta',
                 'buffer_batch': '5',
                 'total buffer': Bt,
                 'total ligand': Lt,
                 'pH': 7.15})
    filename = os.path.join(d, '160722 GCaMP6s 37C buffered set1.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            refolded=True,
                            datadir=datadir,
                            experiment_date='2016-07-22',
                            **info)
    experiments.append(exp)
    filename = os.path.join(d, '160722 GCaMP6s 37C buffered set2.csv')
    exp = ivd.import_ivdata(filename, 'spectra',
                            refolded=True,
                            datadir=datadir,
                            experiment_date='2016-07-22',
                            **info)
    experiments.append(exp)

    return experiments


def import_itcdata(datadir, warn_missing=False, warn_incomplete=False):
    '''
    all concentrations with be stored in M.
    '''
    n_fields = ['cL_M']
    u_fields = ['V_inj_dir', 'V_inj0_dir', 'V_inj_rev', 'V_inj0_rev',
                'cL_0', 'cM_0', 'cL_0']
    m_fields = ['V_cell_dir', 'V_cell_rev']

    stringvals = ['name', 'macromolecule', 'ligand', 'performed_by', 'stock']
    lowercasevals = ['stock', 'performed_by']
    boolvals = ['refolded', 'reference_subtraction']
    skipvals = ['-', '']

    experiments = []
    itcdir = os.path.join(datadir, 'ITC', 'integrated_signal_nitpic')
    configfile = os.path.join(itcdir, 'experiment_parameters.txt')
    raw_config = ivd.read_itc_configtxt(configfile)
    valnames = raw_config[0]

    # FIXME include instrument
    info = standardinfo()
    info.update({'data units': '$\\mu$cal'})
    del info['ligand']  # we'll read it from the info file
    del info['macromolecule']  # we'll read it from the info file

    for i in range(1, len(raw_config)):
        row = raw_config[i]
        e = dict()

        if np.any([c == '-' for c in row]):
            if warn_incomplete:
                warnings.warn("skipping row {0}, incomplete".format(i))
            continue

        v = dict()

        for j, name in enumerate(valnames):
            if name in skipvals:
                continue
            elif name in stringvals:
                v[name] = row[j]
                if name in lowercasevals:
                    v[name] = v[name].lower()
            elif name in boolvals:
                if row[j].isdigit():  # note that bool('0') == True
                    v[name] = bool(float(row[j]))
                elif isinstance(row[j], str):
                    if row[j] == 'True':
                        v[name] = True
                    elif row[j] == 'False':
                        v[name] = False
                    else:
                        raise ValueError("Invalid boolean value")
                else:
                    v[name] = bool(row[j])
            else:
                v[valnames[j]] = float(row[j])
                if valnames[j] in n_fields:
                    v[valnames[j]] /= 1e9
                elif valnames[j] in u_fields:
                    v[valnames[j]] /= 1e6
                elif valnames[j] in m_fields:
                    v[valnames[j]] /= 1e3

        e['temperature'] = v['T']
        e['macromolecule_concentration'] = v['cM_0']  # nominal, inaccurate
        e['total_ligand_other_solution'] = v['cL_0']  # nominal, accurate
        e['total_ligand_in_macromolecule'] = v['cL_M']  # nominal, inaccurate
        e['quality'] = v['quality']
        e['macromolecule'] = v['macromolecule']
        e['ligand'] = v['ligand']
        e['refolded'] = v['refolded']
        e['performed_by'] = v['performed_by']
        e['stock'] = v['stock']
        e['reference corrected'] = v['reference_subtraction']
        e['protein_purification'] = v['stock']

        datafile = os.path.join(itcdir, v['name']) + '.dat'
        if not os.path.isfile(datafile):
            if warn_missing:
                warnings.warn("file not found: {0}".format(datafile))
            continue

        e['data'] = ivd.read_nitpic_datfile(datafile)
        ninjections = e['data'].size  # doesn't include first mini-injection
        '''
        the field V_other indicates the total volume of solution in the ITC
        cell from the solution that does NOT contain macromolecule at each step
        of the experiment, while V_M is the total volume of the macromolecule-
        containing solution. both solutions contain calcium, though generally
        the concentration will be high and known in the "other" solution, but
        low and unknown in the macromolecule solution
        '''
        if v['V_cell_dir'] != 0.0:  # "forward titration"
            assert v['V_cell_rev'] == 0.0 and v['V_inj_rev'] == 0.0 and \
                v['V_inj0_rev'] == 0.0 and v['V_inj_dir'] != 0.0 and \
                v['V_inj0_dir'] != 0.0, "invalid settings"
            e['titration type'] = 'ligand into macromolecule'

            e['V_M'] = np.full(ninjections + 1, v['V_cell_dir'], dtype=float)

            # calculate total injected volume after each injection
            e['V_other'] = v['V_inj0_dir'] + \
                np.arange(ninjections + 1, dtype=float) * v['V_inj_dir']

        else:  # "reverse titration"
            assert v['V_cell_rev'] != 0.0 and v['V_inj_rev'] != 0.0 and \
                v['V_inj0_rev'] != 0.0 and v['V_inj_dir'] == 0.0 and \
                v['V_inj0_dir'] == 0.0, "invalid settings"
            e['titration type'] = 'macromolecule into ligand'

            e['V_other'] = np.full(ninjections + 1, v['V_cell_rev'])

            # calculate total injected volume after each injection
            e['V_M'] = v['V_inj0_rev'] + \
                np.arange(ninjections + 1, dtype=float) * v['V_inj_rev']

        experiments.append(ivd.import_ivdata(datafile, 'itc',
                                             datadir=datadir,
                                             **e,
                                             **info))

    return experiments


def import_all(datadir):
    specdata = import_specdata(datadir)
    itcdata = import_itcdata(datadir)
    stoppedflowdata = import_stoppedflowdata(datadir)

    return specdata + itcdata + stoppedflowdata


if __name__ == '__main__':  # running as script
    datadir = ivd.autodatadir()
    experiments = import_all(datadir)
