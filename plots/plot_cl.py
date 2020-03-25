import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
import healpy as hp
from astropy.io import fits
from matplotlib import rc
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
rc('text', usetex=True)

cosmo = ccl.Cosmology(Omega_c=0.26066676,
                      Omega_b=0.048974682,
                      h=0.6766,
                      sigma8=0.8102,
                      n_s=0.9665)

dat = fits.open("../data/hetdex_optical_ids.fits")[1].data
msk_d = hp.read_map(f"../outputs/mask_d_256.fits", verbose=False).astype(bool)
ipix = hp.ang2pix(256, dat['RA'], dat['DEC'], lonlat=True)
dat = dat[(dat['Total_flux'] >= 2) & msk_d[ipix]]
nz, z = np.histogram(dat['z_best'], bins=40, range=[0, 4], density=True)
z = 0.5 * (z[1:] + z[:-1])
nzb = np.zeros(len(nz)+1)
nzb[1:] = nz
nz = nzb
zb = np.zeros(len(z)+1)
zb[1:] = z
z = zb

t = ccl.NumberCountsTracer(cosmo, False, (z, nz), (z, np.ones_like(z)))
tb = ccl.NumberCountsTracer(cosmo, False, (z, nz),
                            (z, 1.3/ccl.growth_factor(cosmo, 1./(1+z))))
ls = np.load("../outputs/cl_vac_Itotal2.000_mean_lores_mask_cls.npz")['ls']
cls = ccl.angular_cl(cosmo, t, t, ls)
cls_b = ccl.angular_cl(cosmo, tb, tb, ls)

data = {}
data['fiducial'] = {}
data['fiducial']['name'] = 'Fiducial'
data['fiducial']['fmt'] = 'ko'
data['fiducial']['data'] = np.load("../outputs/cl_vac_Itotal2.000_mean_lores_mask_cls.npz")
data['radio'] = {}
data['radio']['name'] = 'Radio catalog'
data['radio']['fmt'] = 'r.'
data['radio']['data'] = np.load("../outputs/cl_rad_Itotal2.000_mean_lores_mask_cls.npz")
data['median'] = {}
data['median']['name'] = 'Mask from median'
data['median']['fmt'] = 'b.'
data['median']['data'] = np.load("../outputs/cl_vac_Itotal2.000_median_lores_mask_cls.npz")
data['hires'] = {}
data['hires']['name'] = 'High-res mask'
data['hires']['fmt'] = 'y.'
data['hires']['data'] = np.load("../outputs/cl_vac_Itotal2.000_mean_hires_mask_cls.npz")
data['peak'] = {}
data['peak']['name'] = 'Cut on peak flux'
data['peak']['fmt'] = 'g.'
data['peak']['data'] = np.load("../outputs/cl_vac_Ipeak2.000_mean_lores_mask_cls.npz")

plt.figure()
for _, di in data.items():
    d = di['data']
    plt.errorbar(d['ls'], d['cls'] - d['nls'], yerr=np.sqrt(np.diag(d['cov'])),
                 fmt=di['fmt'], label=di['name'])
plt.loglog()
for b in [1.5]:
    plt.plot(ls, b**2 * cls, label=r'$b=%.1lf$' % b, lw=2)
plt.plot(ls, cls_b, '--', label=r'$b=1.3/D(z)$')
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$C_\ell$', fontsize=16)
plt.legend(loc='lower left', ncol=2, fontsize=12, labelspacing=0.1)
plt.gca().tick_params(labelsize="large")
plt.savefig("cl_gg.pdf", bbox_inches='tight')
plt.show()
