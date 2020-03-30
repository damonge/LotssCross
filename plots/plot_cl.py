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

t1 = ccl.NumberCountsTracer(cosmo, False, (z, nz), (z, np.ones_like(z)))
t2 = ccl.NumberCountsTracer(cosmo, False, (z, nz),
                            (z, 1./ccl.growth_factor(cosmo, 1./(1+z))))
ls = np.load("../outputs/cl_vac_Itotal2.000_mean_lores_mask_cls.npz")['ls']
cls1 = ccl.angular_cl(cosmo, t1, t1, ls)
cls2 = ccl.angular_cl(cosmo, t2, t2, ls)

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
ind_use = ls < 512
def get_b2(dd, cls_t, inds):
    from scipy.stats import chi2
    v_d = (dd['cls'] - dd['nls'])[inds]
    v_t = cls_t[inds]
    icv = np.linalg.inv(dd['cov'][inds][:, inds])
    var_b2 = 1./np.dot(v_t, np.dot(icv, v_t))
    b2 = np.dot(v_d, np.dot(icv, v_t)) * var_b2
    b = np.sqrt(b2)
    eb = 0.5 * np.sqrt(var_b2 / b2)
    rs = v_d - b2 * v_t
    chi_2 = np.dot(rs, np.dot(icv, rs))
    chi_2_r = chi_2 / (len(rs)-1)
    pte = 1 - chi2.cdf(chi_2, len(rs) - 1)
    print(f"b = {b} +- {eb}, chi2/dof = {chi_2_r}, p = {pte}, dof = %d" % (len(rs) - 1))
    plt.errorbar(ls[inds], rs / np.sqrt(np.diag(dd['cov']))[inds],
                 yerr=1., fmt='.')
    return b, eb, chi_2, pte
b1, eb1, chi21, pte1 = get_b2(data['fiducial']['data'], cls1, ind_use)
b2, eb2, chi22, pte2 = get_b2(data['fiducial']['data'], cls2, ind_use)
plt.plot(ls[ind_use], np.zeros(np.sum(ind_use)), 'k--', lw=1)

plt.figure()
for _, di in data.items():
    d = di['data']
    plt.errorbar(d['ls'], d['cls'] - d['nls'], yerr=np.sqrt(np.diag(d['cov'])),
                 fmt=di['fmt'], label=di['name'])
plt.loglog()
plt.plot(ls, b1**2 * cls1, label=r'$b=%.1lf$' % b1, lw=2)
plt.plot(ls, b2**2 * cls2, '--', label=r'$b=%.1lf/D(z)$' % b2)
plt.xlabel(r'$\ell$', fontsize=16)
plt.ylabel(r'$C_\ell$', fontsize=16)
plt.legend(loc='lower left', ncol=2, fontsize=12, labelspacing=0.1)
plt.gca().tick_params(labelsize="large")
plt.savefig("cl_gg.pdf", bbox_inches='tight')
plt.show()
