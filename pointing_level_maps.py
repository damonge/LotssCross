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
npoint_all = np.zeros(npix_hi)
npoint_good = np.zeros(npix_hi)
ivar_all = np.zeros(npix_hi)
ivar_good = np.zeros(npix_hi)
for n in pt.pointings['name']:
    print(n)
    fsk, mp = pt.get_pointing_map(n)
    pix_ids = fsk.pos2pix(ra, dec)
    points_in_frame = pix_ids >= 0
    is_good = np.zeros(npix_hi, dtype=bool)
    is_good[points_in_frame] = ~np.isnan(mp[pix_ids[points_in_frame]])
    inv_noivar = 1./np.std(mp[~np.isnan(mp)])**2
    npoint_all[is_good] += 1
    ivar_all[is_good] += inv_noivar
    if n not in pt.bad_pointings['name']:
        npoint_good[is_good] += 1
        ivar_good[is_good] += inv_noivar

# Make total mask and write to file
msk_all = np.zeros(npix_hi, dtype=bool)
msk_all[npoint_all > 0] = 1
msk_good = np.zeros(npix_hi, dtype=bool)
msk_good[npoint_good > 0] = 1

hp.write_map(pt.prefix_out + f'hp{nside_hi}_mask.fits.gz',
             msk_all, overwrite=True)
hp.write_map(pt.prefix_out + f'hp{nside_hi}_npoint.fits.gz',
             npoint_all, overwrite=True)
hp.write_map(pt.prefix_out + f'hp{nside_hi}_ivar.fits.gz',
             ivar_all, overwrite=True)
hp.write_map(pt.prefix_out + f'hp{nside_hi}_mask_good.fits.gz',
             msk_good, overwrite=True)
hp.write_map(pt.prefix_out + f'hp{nside_hi}_ivar_good.fits.gz',
             ivar_good, overwrite=True)
hp.write_map(pt.prefix_out + f'hp{nside_hi}_npoint_good.fits.gz',
             npoint_good, overwrite=True)

# Plot
plot_lotss_map(npoint_all)
plot_lotss_map(msk_all)
plot_lotss_map(ivar_all)
plot_lotss_map(npoint_good)
plot_lotss_map(msk_good)
plot_lotss_map(ivar_good)
plt.show()
