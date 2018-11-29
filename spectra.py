'''modeling of fluorescence spectroscopy experiments'''
import copy
import numpy as np
import names
import modelfit as mf
import titration
import util


def spectra_scale_name(exp):
    '''name of data scale for spectra'''
    if 'instrument' not in exp.keys():
        return None
    assert '*' not in exp['instrument'], "illegal character"
    scale_name = 'spectrometer {0}, {1}'.format(exp['instrument'],
                                                exp['spectra type'])
    if 'slit width (nm)' in exp.keys():
        scale_name += ', {0} nm slit width'.format(exp['slit width (nm)'])
    if 'lamp power (W)' in exp.keys():
        power = int(np.round(exp['lamp power (W)']))
        scale_name += ', {0} W'.format(power)
    if 'protein_purification' in exp.keys():
        scale_name += ' ({0})'.format(exp['protein_purification'])
        '''logic is that different pools might have different brightnesses
        due to different rates of correct fluorophore formation. this does not
        yet take into account misfolding that could cause changes in binding'''
    return scale_name


def spectra2sbmodel(exp,
                    nbindingsteps,
                    min_wavelength=0.0,
                    max_wavelength=np.inf,
                    saturation_correction='none',
                    **opts):
    '''
    note that we're using the same regressor names for buffered and unbuffered
    titrations. this is implicitly assigning a total protein concentration of 1
    to the buffered titrations. this is ok because we will have scalefactors
    available when doing global fits
    '''
    opts.setdefault('concentration_unit', 1e-6)
    exp = copy.deepcopy(exp)

    # general titration stuff:
    p0, parameter_names, param_bounds, extra_inputs = \
        titration.titration_params(exp, nbindingsteps, **opts)

    # spectra-specific stuff:
    regressor_names = names.bindingstatenames(exp, nbindingsteps)
    nregressors = nbindingsteps + 1

    if exp['spectra type'] == 'fluorescence':
        wav = exp['excitation wavelengths']
    else:  # absorption
        wav = exp['wavelengths']

    if exp['spectra type'] == 'fluorescence':
        wavok = (wav >= min_wavelength) & (wav <= max_wavelength)
    else:  # fixme support absorption too w/ different optval
        wavok = np.ones_like(wav, dtype=bool)
    wav = wav[wavok]
    assert wav.size > 0, "no valid wavelengths for {0}" \
                             .format(exp['basename'])
    if np.all(wav == np.round(wav)):
        # don't display decimal points if all wavelengths are integers
        wav = np.round(wav)

    # fixme support reference correction etc.
    if exp['spectra type'] == 'fluorescence':
        data_types = ['fluorescence (\\lambda_{{ex}}={0} nm)'
                      .format(util.roundifint(w)) for w in wav]
    else:  # absorption
        data_types = ['optical density ({0} nm)'
                      .format(util.roundifint(w)) for w in wav]

    data = exp['data'][wavok, :].T  # wavelengths are now columns
    sbg = 0.0
    if exp['background present']:
        sbg = exp['background spectrum'][wavok].T
    if saturation_correction == 'poisson':
        '''
        forward model is that for photon rate lambda and sampling rate maxval,
        the probability of a photon detection is simply one minus the pdf of the
        Poisson distribution at zero: 1 - exp(-lambda / maxval). Multiplying
        this by the sampling rate gives the expected overall rate. We invert the
        formula to retrieve lambda from the data values
        '''
        assert exp['spectra type'] == 'fluorescence', \
            "correction for fluorescence spectra only"
        maxval = exp['maximum data value']
        assert (data > 0).all() and (data < maxval).all(), \
            "invalid photon counts"
        p_photon = data / maxval  # probability of detecting a photon
        p_photon_bg = sbg / maxval
        data = maxval * np.log(1.0 / (1.0 - p_photon))
        sbg = maxval * np.log(1.0 / (1.0 - p_photon_bg))
    elif saturation_correction != 'none':
        raise ValueError("unknown saturation correction: {0}"
                         .format(saturation_correction))
    data -= sbg

    coef_bounds = (np.zeros((nregressors, wav.size)),
                   np.full((nregressors, wav.size), np.inf))
    '''
    fluorescence data which is averaged over longer integration times or more
    repeats will have smaller residuals. in order to fit multiple data with a
    single variance parameter, we therefore rescale each experiment's data so
    that the residuals will be the same size, effectively putting greater
    weight on experiments with smaller expected noise. we can only do this when
    we have the relevant info in the experiment dict.
    '''
    noise_info_fields = ['integration time (ms)',
                         'n_averaged',
                         'instrument',
                         'emission wavelength']
    if np.all([f in exp.keys() for f in noise_info_fields]):
        noise_type = '{0} noise, \\lambda_{{em}} = {1}'.format(
            exp['instrument'], exp['emission wavelength'])
        noise_factor = np.sqrt((exp['integration time (ms)'] / 1000.0) *
                               exp['n_averaged'])
    else:
        noise_type = 'spectral noise, unknown settings@'
        noise_factor = 1.0

    model = mf.regressionmodel(data,
                               titration.regressorfcn_titration,
                               p0,
                               regressorjac=titration.regressorjac_titration,
                               parameter_names=parameter_names,
                               regressor_names=regressor_names,
                               data_types=data_types,
                               scale_name=spectra_scale_name(exp),
                               noise_type=noise_type,
                               param_bounds=param_bounds,
                               coef_bounds=coef_bounds,
                               extra_inputs=extra_inputs,
                               xvals=wav)

    model.concentration_unit = opts['concentration_unit']
    model.noise_factor = noise_factor

    return model
