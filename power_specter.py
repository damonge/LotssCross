import numpy as np
import matplotlib.pyplot as plt
import healpy as hp
from astropy.io import fits
from optparse import OptionParser
import pymaster as nmt
from utils import plot_lotss_map
import os


parser = OptionParser()
parser.add_option('--nside', dest='nside', default=256, type=int,
                  help='Nside parameter, default=256')
parser.add_option('--use-median', dest='use_median', default=False,
                  action='store_true',
                  help='Use median rms (instead of mean)')
parser.add_option('--Ithr', dest='I_thr', default=0.,  type=float,
                  help='Flux threshold (default: 0)')
parser.add_option('--snr', dest='snr', default=5,  type=int,
                  help='S/N threshold (default: 5)')
parser.add_option('--cut-peak', dest='cut_peak', default=False,
                  action='store_true',
                  help='Use peak flux? (default: False)')
parser.add_option('--use-vac', dest='use_vac', default=False,
                  action='store_true',
                  help='Use value-added catalog? (default: False)')
parser.add_option('--use-hires-mask', dest='use_hires_mask', default=False,
                  action='store_true',
                  help='Use high-resoluion mask? (default: False)')
parser.add_option('--recompute-mcm', dest='recompute_mcm', default=False,
                  action='store_true',
                  help='Recompute MCM? (default: False)')
parser.add_option('--plot-stuff', dest='plot_stuff', default=False,
                  action='store_true',
                  help='Make plots? (default: False)')
(o, args) = parser.parse_args()


npix = hp.nside2npix(o.nside)
output_prefix = 'outputs/cl_'

# Read catalog
if o.use_vac:
    fname_cat = "data/radio_catalog.fits"
    output_prefix += 'vac_'
else:
    fname_cat = "data/hetdex_optical_ids.fits"
    output_prefix += 'rad_'
cat = fits.open(fname_cat)[1].data

# Flux cut
if o.cut_peak:
    flux_flag = 'Peak_flux'
    output_prefix += 'Ipeak%.3lf_' % o.I_thr
else:
    flux_flag = 'Total_flux'
    output_prefix += 'Itotal%.3lf_' % o.I_thr
cat = cat[cat[flux_flag] >= o.I_thr]

# Mask
# 1. p-map
if o.use_median:
    meanmed = 'median'
else:
    meanmed = 'mean'
output_prefix += meanmed + '_'
fname_p = "outputs/p_map_rms_" + meanmed + ("_Imin%.3lf.fits" % o.I_thr)
p_map = hp.read_map(fname_p, verbose=False)
p_map /= np.amax(p_map)
# 2. Footprint mask
if o.use_hires_mask:
    msk_b = hp.ud_grade(hp.read_map("outputs/pointings_hp2048_mask_good.fits.gz",
                                    verbose=False).astype(float),
                        nside_out=o.nside)
    output_prefix += 'hires_mask_'
else:
    msk_b = hp.read_map("outputs/mask_d_256.fits",
                        verbose=False).astype(float)
    output_prefix += 'lores_mask_'
# 3. Masked fraction
masked_fraction = p_map * msk_b
masked_fraction[masked_fraction < 0.5] = 0
print(output_prefix)

# Map
ipix = hp.ang2pix(o.nside, cat['RA'], cat['DEC'], lonlat=True)
map_n = np.bincount(ipix, minlength=npix).astype(float)
good_pix = masked_fraction > 0.
mean_n = np.sum(map_n[good_pix]) / np.sum(masked_fraction[good_pix])
map_d = np.zeros(npix)
map_d[good_pix] = map_n[good_pix] / (masked_fraction[good_pix] * mean_n) - 1


# Power spectrum
b = nmt.NmtBin(o.nside, nlb=50)
f = nmt.NmtField(masked_fraction, [map_d])
w = nmt.NmtWorkspace()
if os.path.isfile(output_prefix+'mcm.fits') and not o.recompute_mcm:
    w.read_from(output_prefix+'mcm.fits')
else:
    print("Computing MCM")
    w.compute_coupling_matrix(f, f, b)
    w.write_to(output_prefix+'mcm.fits')
l_eff = b.get_effective_ells()
n_bpw = len(l_eff)
cl = w.decouple_cell(nmt.compute_coupled_cell(f, f))[0]

# Shot noise
n_dens = mean_n * npix / (4 * np.pi)
nl_coupled = np.ones(3 * o.nside) * np.mean(masked_fraction) / n_dens
nl = w.decouple_cell([nl_coupled])[0]

# Covariance
# 1. Theory power spectra
l_arr = np.arange(3 * o.nside)
alpha_fit = np.log((cl-nl)[-1] / (cl-nl)[0]) / np.log(l_eff[-1] / l_eff[0])
if np.isnan(alpha_fit):
    alpha_fit = -1.3
print(alpha_fit)
cl_th = (cl-nl)[n_bpw // 3] * \
        ((l_arr+10) / l_eff[n_bpw // 3])**alpha_fit + \
        np.mean(nl)
# 2. MCM
cw = nmt.NmtCovarianceWorkspace()
if os.path.isfile(output_prefix+'cmcm.fits') and not o.recompute_mcm:
    cw.read_from(output_prefix+'cmcm.fits')
else:
    print("Computing CMCM")
    cw.compute_coupling_coefficients(f, f)
    cw.write_to(output_prefix+'cmcm.fits')
cov = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                              [cl_th], [cl_th], [cl_th], [cl_th],
                              w, wb=w)
err = np.sqrt(np.diag(cov))


# Write output
np.savez(output_prefix+'cls', ls=l_eff, cls=cl, cov=cov,
         nls=nl, l_th=l_arr, cl_th=cl_th)


# Plotting
if o.plot_stuff:
    plt.figure()
    plt.errorbar(l_eff, cl - nl, yerr=err, fmt='r.',
                 label=r'Data, $I<%.1lf {\rm mJy}$' % o.I_thr)
    plt.plot(l_arr, cl_th - np.mean(nl), 'k-', label='Power-law fit')
    plt.plot(l_eff, nl, 'g--', label='Shot noise prediction')
    plt.loglog()
    plt.xlim([0.9*l_eff[0], 1.1*l_eff[-1]])
    plt.ylim([0.5*np.amin((cl-nl)-err),
              2*np.amax((cl-nl)+err)])
    plt.xlabel(r'$\ell$', fontsize=14)
    plt.ylabel(r'$C_\ell$', fontsize=14)
    plt.legend(loc='upper right')

    plot_lotss_map(map_n, title='Counts')
    plot_lotss_map(map_d, title=r'$\delta_g$')
    plot_lotss_map(masked_fraction, title='Masked fraction')
    plt.show()
