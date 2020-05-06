import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import utils as ut
from matplotlib import rc
from astropy.io import fits
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica'],
              'size': 15})
rc('text', usetex=True)

nside = 2048
npix = hp.nside2npix(nside)

# 2 mJy depth
p_map2 = hp.read_map("outputs/p_map_rms_median_Imin2.000.fits",
                     dtype=None, verbose=False).astype(float)
p_map2 /= np.amax(p_map2)
p_map2 = hp.ud_grade(p_map2, nside_out=nside)

# Binary mask
msk_b = hp.read_map("outputs/pointings_hp2048_mask_good.fits.gz",
                    dtype=None, verbose=False).astype(float)
mask_lofar = p_map2 * msk_b
mask_lofar[mask_lofar < 0.5] = 0

# 0.1 mJy depth
p_map0p1 = hp.read_map("outputs/p_map_rms_median_Imin0.100.fits",
                       dtype=None, verbose=False).astype(float)
p_map0p1 /= np.amax(p_map0p1)
p_map0p1 = hp.ud_grade(p_map0p1, nside_out=nside)

# Noise variance fluctuations
ivar = hp.read_map("outputs/pointings_hp2048_ivar_good.fits.gz",
                   dtype=None, verbose=False).astype(float)
temp_deproj = hp.read_map("outputs/pointings_hp2048_ivar_good.fits.gz",
                          dtype=None, verbose=False).astype(float)
temp_deproj = hp.ud_grade(temp_deproj, nside_out=nside)
temp_mean = np.sum(temp_deproj * mask_lofar) / np.sum(mask_lofar)
temp_deproj = temp_deproj / temp_mean - 1
temp_deproj[mask_lofar <= 0.01] = 0

# Overdensity
cat = fits.open("data/hetdex_optical_ids.fits")[1].data
cat = cat[cat['Total_flux'] >= 2]
ipix = hp.ang2pix(nside, cat['RA'], cat['DEC'], lonlat=True)
map_n = np.bincount(ipix, minlength=npix).astype(float)
ipix_vac = hp.ang2pix(256, cat['RA'], cat['DEC'], lonlat=True)
dat = cat[mask_lofar[ipix] > 0.]
nz, z = np.histogram(dat['z_best'], bins=40, range=[0, 4], density=True)
z = 0.5 * (z[1:] + z[:-1])
nzb = np.zeros(len(nz)+1)
nzb[1:] = nz
nz = nzb
zb = np.zeros(len(z)+1)
zb[1:] = z
z = zb
bz = 1.3
f = ut.Field('lofar_g', 'g', map_n, mask_lofar,
             nz=(z, nz), bz=bz)

# Kappa
mask_planck = hp.read_map('data/mask.fits.gz',
                          dtype=None, verbose=False).astype(float)
alm_planck = hp.read_alm('data/dat_klm.fits')
ll, nll, cll = np.loadtxt('data/nlkk.dat', unpack=True)
ll = ll.astype(int)
cl = np.zeros(ll[-1]+1)
cl[ll[0]:] = cll
nl = np.zeros(ll[-1]+1)
nl[ll[0]:] = nll
wl = (cl-nl)/np.maximum(cl, np.ones_like(cl)*1E-10)
alm_planck = hp.almxfl(alm_planck, wl)
map_planck = hp.alm2map(alm_planck, nside, verbose=False)


ut.plot_lotss_map(map_planck*mask_lofar*mask_planck,
                  mask=mask_lofar*mask_planck,
                  title=r'$\kappa$',
                  fname='plots/kappa.pdf')
ut.plot_lotss_map(p_map0p1, mask=mask_lofar,
                  title=r'Depth, $I_{\rm cut}=0.1\,{\rm MJy}$',
                  fname='plots/depth_0p1.pdf')
ut.plot_lotss_map(p_map2, mask=mask_lofar,
                  title=r'Depth, $I_{\rm cut}=2\,{\rm MJy}$',
                  fname='plots/depth_2p0.pdf')
ut.plot_lotss_map(temp_deproj, mask=mask_lofar,
                  title=r'Pointing noise variations',
                  fname='plots/ivar.pdf')
ut.plot_lotss_map(f.msk, title=r'LOFAR mask',
                  fname='plots/mask.pdf')
ut.plot_lotss_map(f.mp, mask=mask_lofar,
                  title=r'$\delta_g$',
                  fname='plots/delta_g.pdf')
plt.show()
