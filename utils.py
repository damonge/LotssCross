import wget
import os
import sys
import tarfile
import matplotlib.pyplot as plt
import numpy as np
import healpy as hp
import flatmaps as fm
from scipy.special import erf
from astropy.io import fits
import pyccl as ccl
import pymaster as nmt


def get_default_cosmo():
    return ccl.Cosmology(Omega_c=0.26066676,
                         Omega_b=0.048974682,
                         h=0.6766,
                         sigma8=0.8102,
                         n_s=0.9665)


class Field(object):
    def __init__(self, name, kind, mp, msk, nz=None, cosmo=None, bz=None):
        self.name = name
        if kind not in ['g', 'k']:
            raise ValueError("Field types are 'g' and 'k'")
        self.kind = kind
        self.mp = mp
        self.npix = len(mp)
        self.nside = hp.npix2nside(self.npix)
        if len(msk) != self.npix:
            raise ValueError("Map and mask must have the same pixelization")
        self.msk = msk
        if (kind == 'g') and (nz is None):
            raise ValueError("N(z) needed for galaxy clustering")
        if self.kind == 'g':
            # N(z) and b(z)
            self.z, self.nz = nz
            if bz is None:
                self.bz = np.ones_like(self.z)
            else:
                if np.ndim(bz) == 0:
                    self.bz = np.ones_like(self.z) * bz
                else:
                    self.bz = bz

            # Delta map
            good_pix = self.msk > 0.
            self.mean_n = np.sum(self.mp[good_pix])/np.sum(self.msk[good_pix])
            d = np.zeros(self.npix)
            d[good_pix] = self.mp[good_pix]/(self.msk[good_pix]*self.mean_n)-1
            self.mp = d

            # Mode-coupled noise spectrum
            n_dens = self.mean_n*self.npix/(4*np.pi)
            nl = np.mean(self.msk) / n_dens
            self.nl_coupled = np.ones(3*self.nside) * nl

        if cosmo is None:
            cosmo = get_default_cosmo()
        self.t = self._get_tracer(cosmo)
        self.f = self._get_nmtfield()

    def _get_nmtfield(self):
        return nmt.NmtField(self.msk, [self.mp], n_iter=0)

    def _get_tracer(self, cosmo):
        if self.kind == 'g':
            t = ccl.NumberCountsTracer(cosmo, False, (self.z, self.nz),
                                       (self.z, self.bz))
        elif self.kind == 'k':
            t = ccl.CMBLensingTracer(cosmo, z_source=1100.)
        else:
            raise ValueError("'kind' must be 'g' or 'k'")
        return t



class Pointings(object):
    def __init__(self, fname="data/pointings.txt",
                 fname_bad="data/bad_pointings.txt",
                 prefix_data='data/res_low_',
                 prefix_out='outputs/pointings_'):
        self.pointings = self._get_pointing_names(fname)
        self.bad_pointings = self._get_pointing_names(fname_bad)
        self.prefix_data = prefix_data
        self.prefix_out = prefix_out

    def _get_pointing_names(self, fname):
        point_list = np.genfromtxt(fname, dtype=[('name', 'U16'), ('RA', 'f8'), ('DEC', 'f8')])
        return point_list

    def download_pointings(self, names):
        for n in names:
            url_pre = "https://lofar-surveys.org/public/all-residual-mosaics/"
            url = url_pre + n + "-low-mosaic.fits"
            print(f"  {n}")
            dwl_file(url, self.prefix_data + n+'.fits',
                     verbose=False)

    def get_pointing_map(self, name):
        fsk, mp = fm.read_flat_map(self.prefix_data + name + '.fits')
        return fsk, mp

    def get_pointing_mask_from_coords(self, name, ra, dec,
                                      recompute=False):
        nside = hp.npix2nside(len(ra))
        fname_out = self.prefix_out + name + f'_hp{nside}_mask.fits.gz'
        if (not os.path.isfile(fname_out)) or recompute:
            print(f"Computing mask for {name}")
            fsk, mp_flat = self.get_pointing_map(name)
            pix_ids = fsk.pos2pix(ra, dec)
            pix_in = pix_ids > 0
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
        wget.download(url, out=fname_out, bar=None)  # bar=verbose
    else:
        if verbose:
            print(f"Found {fname_out}")


def untar_file(fname, path='.'):
    if tarfile.is_tarfile(fname):
        tar = tarfile.open(fname)
        tar.extractall(path=path)
        tar.close()
        print(f'Extracted {fname} in {path}')
    else:
        print(f'{fname} is not a tar file')


class FluxPDF(object):
    def __init__(self, fname_in="data/skads_flux_counts.result"):
        from scipy.interpolate import interp1d
        # Read flux distribution from SKADS' S3-SEX simulation
        self.log_flux, counts = np.loadtxt(fname_in, unpack=True,
                                           delimiter=',', skiprows=1)
        self.log_flux += 3  # Use mJy instead of Jy
        # Assuming equal spacing
        self.dlog_flux = np.mean(np.diff(self.log_flux))
        self.log_flux = self.log_flux[counts >= 0]
        counts = counts[counts >= 0]
        self.probs = counts / np.sum(counts)
        # Cut to non-zero counts
        self.lpdf = interp1d(self.log_flux, np.log10(counts),
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

    def compute_p_map(self, q, std_map, Imin, alpha=-0.7):
        lf = self.log_flux + alpha * np.log10(144. / 1400.)
        p_map = np.zeros(len(std_map))
        for ip, std in enumerate(std_map):
            if std > 0:
                Ithr = max(q * std, Imin)
                x = (Ithr - 10.**lf) / (np.sqrt(2.) * std)
                comp = 0.5 * (1 - erf(x))
                p_map[ip] = np.sum(self.probs * comp)
        return p_map

    def draw_random_fluxes(self, n, alpha=-0.7, lf_thr_low=-3.5):
        msk = self.log_flux >= lf_thr_low
        lf_ax = self.log_flux[msk]
        p_ax = self.probs[msk]
        p_ax /= np.sum(p_ax)
        lf = np.random.choice(lf_ax, size=n, p=p_ax)
        lf += self.dlog_flux * (np.random.random(n)-0.5)
        # Extrapolate to 144 MHz
        # Assumption: I_nu = I_1400 * (nu / 1400)^alpha
        if alpha != 0:
            lf += alpha * np.log10(144. / 1400.)
        return lf


def plot_lotss_map(mp, **kwargs):
    hp.cartview(mp, lonra=[159.5, 232.5], latra=[44, 59],
                **kwargs)


def get_random_positions(n):
    c0 = np.cos(np.radians(90 - 44.5))
    c1 = np.cos(np.radians(90 - 58.5))
    ra = 160. + (232. - 160.) * np.random.random(n)
    dec = 90 - np.degrees(np.arccos(c0 + (c1 - c0) * np.random.random(n)))
    return ra, dec


def print_info_fits(fname):
    """ Print on screen fits file info.

    Args:
        fname: path of the input file.

    Returns:
        None

    """

    with fits.open(fname) as hdul:
        print(hdul.info())
        sys.stdout.flush()
    return


def read_header_from_fits(fname, name):
    """ Open a fits file and read header from it.

    Args:
        fname: path of the data file.
        name: name of the data we want to extract.

    Returns:
        header.

    """
    with fits.open(fname) as fn:
        return fn[name].header


def read_from_fits(fname, name):
    """ Open a fits file and read data from it.

    Args:
        fname: path of the data file.
        name: name of the data we want to extract.

    Returns:
        array with data for name.

    """
    with fits.open(fname) as fn:
        return fn[name].data


def write_to_fits(fname, array, name, type='image', header=None):
    """ Write an array to a fits file.

    Args:
        fname: path of the input file.
        array: array to save.
        name: name of the image.

    Returns:
        None

    """

    # If file does not exist, create it
    if not os.path.exists(fname):
        hdul = fits.HDUList([fits.PrimaryHDU()])
        hdul.writeto(fname)
    # Open the file
    with fits.open(fname, mode='update') as hdul:
        try:
            hdul.__delitem__(name)
        except KeyError:
            pass
        if type == 'image':
            hdul.append(fits.ImageHDU(array, name=name, header=header))
        elif type == 'table':
            hdul.append(array)
        else:
            print('Type '+type+' not recognized! Data not saved to file!')
            return True
    print('Appended ' + name.upper() + ' to ' + os.path.relpath(fname))
    sys.stdout.flush()
    return
