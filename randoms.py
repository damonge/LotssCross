import numpy as np
import healpy as hp
from utils import FluxPDF, get_random_positions, plot_lotss_map
import matplotlib.pyplot as plt
from astropy.io import fits


# Read mask and rms map
mask = hp.read_map("outputs/pointings_hp2048_mask.fits.gz",
                   verbose=False).astype(bool)
npix_hi = len(mask)
nside_hi = hp.npix2nside(npix_hi)

map_rms = hp.read_map("outputs/map_rms_mean_256.fits",
                      verbose=False)
npix_lo = len(map_rms)
nside_lo = hp.npix2nside(npix_lo)
mask_lo = hp.ud_grade(mask, nside_out=nside_lo)
# We need to remove areas where the RMS is zero
mask_lo[map_rms <= 0] = False

fpdf = FluxPDF()


def get_random_batch(npoint):
    # Generate random points over the full map
    print(f"Onset : {npoint}")
    ra, dec = get_random_positions(npoint)
    ip_lo = hp.ang2pix(nside_lo, np.radians(90-dec), np.radians(ra))
    ip_hi = hp.ang2pix(nside_hi, np.radians(90-dec), np.radians(ra))

    # Sky mask (low res)
    msk = mask_lo[ip_lo]
    ra = ra[msk]
    dec = dec[msk]
    ip_lo = ip_lo[msk]
    ip_hi = ip_hi[msk]
    npoint = len(ra)
    print(f"Low-res sky mask : {npoint}")

    # Get fluxes
    flux_signal = 10.**fpdf.draw_random_fluxes(npoint)

    # Get noises
    noise_std = map_rms[ip_lo]
    flux = noise_std * np.random.randn(npoint)
    flux += flux_signal

    # Cut in S/N
    msk = flux >= 5 * noise_std
    ra = ra[msk]
    dec = dec[msk]
    ip_lo = ip_lo[msk]
    ip_hi = ip_hi[msk]
    flux = flux[msk]
    flux_signal = flux_signal[msk]
    npoint = len(ra)
    print(f"S/N > 5 : {npoint}")

    # Sky mask (hi-res)
    msk = mask[ip_hi]
    ra = ra[msk]
    dec = dec[msk]
    ip_lo = ip_lo[msk]
    ip_hi = ip_hi[msk]
    flux = flux[msk]
    flux_signal = flux_signal[msk]
    npoint = len(ra)
    print(f"Hi-res sky mask : {npoint}")

    return ra, dec, flux, flux_signal


ra_all = np.array([])
dec_all = np.array([])
flux_all = np.array([])
flux_signal_all = np.array([])

for i in range(200):
    ra, dec, flux, flux_signal = get_random_batch(10000000)
    ra_all = np.concatenate([ra_all, ra])
    dec_all = np.concatenate([dec_all, dec])
    flux_all = np.concatenate([flux_all, flux])
    flux_signal_all = np.concatenate([flux_signal_all, flux_signal])
    print(i, len(ra_all))

hdu = fits.BinTableHDU.from_columns([fits.Column(name='RA',
                                                 format='D',
                                                 array=ra_all),
                                     fits.Column(name='DEC',
                                                 format='D',
                                                 array=dec_all),
                                     fits.Column(name='Flux_true',
                                                 format='D',
                                                 array=flux_signal_all),
                                     fits.Column(name='Flux',
                                                 format='D',
                                                 array=flux_all)])
hdu.writeto("outputs/random.fits", overwrite=True)

plt.figure()
plt.hist(np.log10(flux_signal_all), bins=100,
         density=True, color='b', histtype='step')
plt.hist(np.log10(flux_all), bins=100,
         density=True, color='y', histtype='step')

ip_lo = hp.ang2pix(nside_lo, np.radians(90-dec_all), np.radians(ra_all))
m = np.bincount(ip_lo, minlength=npix_lo)
plot_lotss_map(map_rms)
plot_lotss_map(m)

plt.show()
