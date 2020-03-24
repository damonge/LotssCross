import wget
import os
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import flatmaps as fm


class Pointings(object):
    def __init__(self, fname="data/pointing_names.txt",
                 fname_bad="data/bad_pointings.txt",
                 prefix_data='data/res_low_',
                 prefix_out='outputs/pointings_'):
        self.pointings = self._get_pointing_names(fname)
        self.bad_pointings = self._get_pointing_names(fname_bad)
        self.prefix_data = prefix_data
        self.prefix_out = prefix_out

    def _get_pointing_names(self, fname):
        with open(fname) as f:
            point_list = f.read().split()
        return point_list

    def download_pointings(self, names):
        for n in names:
            url_pre = "https://lofar-surveys.org/public/all-residual-mosaics/"
            url = url_pre + n + "-low-mosaic.fits"
            print(f"  {n}")
            dwl_file(url, self.prefix_data + n+'.fits',
                     verbose=False)

    def get_pointing_map(self, name):
        fsk, mp = fm.read_flat_map(self.prefix_data + name +'.fits')
        return fsk, mp

    def get_pointing_mask_from_coords(self, name, ra, dec,
                                      recompute=False):
        nside = hp.npix2nside(len(ra))
        fname_out = self.prefix_out + name + f'_hp{nside}_mask.fits.gz'
        if (not os.path.isfile(fname_out)) or recompute:
            print(f"Computing mask for {name}")
            fsk, mp_flat = self.get_pointing_map(name)
            pix_ids = fsk.pos2pix(ra, dec)
            pix_in = pix_ids>0
            mp = np.zeros(len(ra), dtype=bool)
            mp[pix_in] = ~np.isnan(mp_flat[pix_ids[pix_in]])
            hp.write_map(fname_out, mp, overwrite=True)
        else:
            print(f"Found {fname_out}")
            mp = hp.read_map(fname_out, verbose=False)
        return mp


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
