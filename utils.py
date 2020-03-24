import wget
import os
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp


def get_pointing_names():
    with open("data/pointing_names.txt") as f:
        point_list = f.read().split()
    return point_list


def dwl_file(url, fname_out, redwl=False, verbose=True):
    if (not os.path.isfile(fname_out)) or redwl:
        if verbose:
            print(f"Downloading {fname_out} from {url}")
        wget.download(url, out=fname_out, bar=verbose)
    else:
        if verbose:
            print(f"Found {fname_out}")


class FluxPDF(object):
    def __init__(self, fname_in="data/skads_flux_counts.result"):
        from scipy.interpolate import interp1d
        # Read flux distribution from SKADS' S3-SEX simulation
        log_flux, counts = np.loadtxt(fname_in, unpack=True,
                                      delimiter=',', skiprows=1)
        log_flux += 3  # Use mJy instead of Jy
        # Cut to non-zero counts
        log_flux = log_flux[counts >= 0]
        counts = counts[counts >= 0]
        self.lpdf = interp1d(log_flux, np.log10(counts),
                             fill_value=-500, bounds_error=False)

    def plot_pdf(self, log_flux_min=-6, log_flux_max=6,
                 n_log_flux=256):
        lf = np.linspace(log_flux_min, log_flux_max, n_log_flux)
        dlf = np.mean(np.diff(lf))
        p = 10.**self.lpdf(lf)
        p /= np.sum(p) * dlf
        plt.figure()
        plt.plot(10.**lf, p, 'k-')
        plt.loglog()
        plt.xlabel(r'$I_{1400}\,{\rm mJy}$', fontsize=14)
        plt.ylabel(r'$dp/d\log_{10}I_{1400}$', fontsize=14)
        plt.show()


def plot_lotss_map(mp, **kwargs):
    hp.cartview(mp, lonra=[155, 236], latra=[40, 62],
                **kwargs)
