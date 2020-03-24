import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from utils import plot_lotss_map, Pointings

# Read pointing data
pt = Pointings()

# Compute pixel positions
nside_hi = 2048
npix_hi = hp.nside2npix(nside_hi)
ra, dec = hp.pix2ang(nside_hi, np.arange(npix_hi), lonlat=True)

# Make per-pointing masks
mp_all = np.zeros(npix_hi)
mp_good = np.zeros(npix_hi)
for n in pt.pointings:
    mp = pt.get_pointing_mask_from_coords(n, ra, dec)
    mp_all += mp
    if n not in pt.bad_pointings:
        mp_good += mp

# Make total mask and write to file
msk = np.zeros(npix_hi, dtype=bool)
msk[mp_all > 0] = 1
msk_good = np.zeros(npix_hi, dtype=bool)
msk_good[mp_good > 0] = 1
hp.write_map(pt.prefix_out + f'hp{nside_hi}_mask.fits.gz',
             msk, overwrite=True)
hp.write_map(pt.prefix_out + f'hp{nside_hi}_npoint.fits.gz',
             mp_all, overwrite=True)
hp.write_map(pt.prefix_out + f'hp{nside_hi}_mask_good.fits.gz',
             msk_good, overwrite=True)
hp.write_map(pt.prefix_out + f'hp{nside_hi}_npoint_good.fits.gz',
             mp_good, overwrite=True)

# Plot
plot_lotss_map(mp_all)
plot_lotss_map(msk)
plot_lotss_map(mp_good)
plot_lotss_map(msk_good)
plt.show()
