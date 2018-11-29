import numpy as np
import warnings


def kd_nM(molecule, pH=7.2, T=37.0, IS_mm=100.0, Ttol=2.0, pHtol=0.5):
    if molecule == 'egta':

        # https://tools.thermofisher.com/content/sfs/manuals/CalciumCalibrationBufferKits_PI.pdf
        # this in turn cites Methods Enzymol 172, 230 (1989);
        data = np.array([
                        [6.50, 3728, 2646],
                        [6.60, 2354, 1672],
                        [6.70, 1487, 1057],
                        [6.75, 1182, 841],
                        [6.80, 940, 669],
                        [6.85, 747, 532],
                        [6.90, 594, 423],
                        [6.95, 472, 337],
                        [7.00, 376, 268],
                        [7.05, 299, 213],
                        [7.10, 238, 170.0],
                        [7.15, 189.1, 135.4],
                        [7.20, 150.5, 107.9],
                        [7.25, 119.8, 86.0],
                        [7.30, 95.4, 68.6],
                        [7.35, 76.0, 54.7],
                        [7.40, 60.5, 43.7],
                        [7.45, 48.2, 34.9],
                        [7.50, 38.5, 27.9],
                        [7.60, 24.5, 17.88],
                        [7.70, 15.61, 11.49],
                        [7.80, 9.99, 7.42],
                        [7.90, 6.41, 4.82],
                        [8.00, 4.13, 3.15],
                        [8.10, 2.68, 2.08],
                        [8.20, 1.75, 1.39]
                        ])
        Tlist = np.array([20., 37.])

        Tlist == np.array([20., 37.])
        Tindex = np.argmin(np.abs(T - Tlist))
        if np.abs(Tlist[Tindex] - T) > Ttol:
            warnings.warn("k_d for {0} not available at T = {1}, using T = {2}"
                          " instead".format(molecule, T, Tlist[Tindex]))

        pHlist = data[:, 0]
        pHindex = np.argmin(np.abs(pHlist - pH))
        if np.abs(pHlist[pHindex] - pH) > pHtol:
            warnings.warn("k_d for {0} not available at pH = {1}, using pH = "
                          "{2} instead".format(molecule, pH, pHlist[Tindex]))

        return data[pHindex, Tindex + 1]

    elif molecule == 'bapta':
        if np.abs(T - 22.0) < Ttol and np.abs(pH - 7.2) < pHtol:
            # https://www.thermofisher.com/de/de/home/references/molecular-probes-the-handbook/tables/ca2-affinities-of-bapta-chelators.html
            # 100 mM KCl, not sure about total IS
            return 160.0  # Maxchelator says 128.5 nM
        elif np.abs(T - 37.0) < Ttol and np.abs(pH - 7.2) < pHtol:
            return 222.8  # Maxchelator, IS 0.15

    else:
        raise ValueError("k_d unavailable for molecule: {0}".format(molecule))


def rates_sec_M(molecule, pH=7.2, T=37.0, IS_mm=100.0, Ttol=2.0, pHtol=0.5):
    if molecule == 'bapta':
        '''
        honestly who knows. Naraghi et al. say 79.0 at 22 deg.
        '''
        koff = 90.0
    else:
        raise ValueError("k_d unavailable for molecule: {0}".format(molecule))
    kd = kd_nM(molecule, pH=pH, T=T, IS_mm=IS_mm, Ttol=Ttol, pHtol=pHtol) / 1e9
    kon = koff / kd

    return koff, kon
