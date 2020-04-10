import os
import sys
import time
import argparse
import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.io import fits
import matplotlib.pyplot as plt
import utils


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--skip_planck', dest='run_planck',
                    default=True, action='store_false',
                    help='Skip Planck pipeline (default: False)')
parser.add_argument('--skip_lofar', dest='run_lofar',
                    default=True, action='store_false',
                    help='Skip Lofar pipeline (default: False)')
parser.add_argument('--nside', type=int, default=2048,
                    help='Nside parameter, default=2048')
parser.add_argument('--path_planck', type=str, default='data/planck/data',
                    help='Path to Planck files, default=data/planck/data')
parser.add_argument('--path_lofar', type=str, default='data',
                    help='Path to Lofar files, default=data')
parser.add_argument('--use_rc', default=False, action='store_true',
                    help='Use radio catalog? (default: False)')
parser.add_argument('--cut_peak', default=False, action='store_true',
                    help='Use peak flux? (default: False)')
parser.add_argument('--I_thr', default=2.,  type=float,
                    help='Flux threshold (default: 2 mJy)')
parser.add_argument('--use_median', default=False, action='store_true',
                    help='Use median rms (instead of mean)')
parser.add_argument('--recompute_mcm', default=False, action='store_true',
                    help='Recompute MCM? (default: False)')
parser.add_argument('--plot-stuff', default=False, action='store_true',
                    help='Make plots? (default: False)')
parser.add_argument('--silent', '-s', dest='verbose', default=True,
                    action='store_false', help='Verbose mode (default: False)')
parser.add_argument('--output', '-o', type=str, default='outputs/cls_cov',
                    help='Output file, default=outputs/cls_cov')
args = parser.parse_args()


# Planck maps
if args.run_planck:
    if args.verbose:
        print('Creating Planck maps')
        sys.stdout.flush()
    time_start = time.time()

    # Load files
    mask_planck = hp.read_map(os.path.join(args.path_planck, 'mask.fits.gz'),
                              dtype=None, verbose=False).astype(float)
    alm_planck = hp.read_alm(os.path.join(args.path_planck, 'dat_klm.fits'))

    # Convert alm to map
    map_planck = hp.sphtfunc.alm2map(alm_planck, args.nside, verbose=False)

    # Upgrade maps
    if hp.get_nside(mask_planck) != args.nside:
        mask_planck = hp.ud_grade(mask_planck, nside_out=args.nside)
    if hp.get_nside(map_planck) != args.nside:
        map_planck = hp.ud_grade(map_planck, nside_out=args.nside)

    if args.verbose:
        print('----> Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()


# Lofar maps
if args.run_lofar:
    if args.verbose:
        print('Creating Lofar maps')
        sys.stdout.flush()
    time_start = time.time()

    # Read catalog
    if args.use_rc:
        fname_cat = os.path.join(args.path_lofar, 'radio_catalog.fits')
    else:
        fname_cat = os.path.join(args.path_lofar, 'hetdex_optical_ids.fits')
    cat = fits.open(fname_cat)[1].data

    # Flux cut
    if args.cut_peak:
        flux_flag = 'Peak_flux'
    else:
        flux_flag = 'Total_flux'
    cat = cat[cat[flux_flag] >= args.I_thr]

    # p-map
    if args.use_median:
        meanmed = 'median'
    else:
        meanmed = 'mean'
    fname_p = "outputs/p_map_rms_" + meanmed + ("_Imin%.3lf.fits" % args.I_thr)
    p_map = hp.read_map(fname_p, dtype=None, verbose=False).astype(float)
    p_map /= np.amax(p_map)
    # Footprint mask
    msk_b = hp.read_map("outputs/mask_d_256.fits", dtype=None,
                        verbose=False).astype(float)
    # Mask
    mask_lofar = p_map * msk_b
    mask_lofar[mask_lofar < 0.5] = 0

    # Upgrade mask
    if hp.get_nside(mask_lofar) != args.nside:
        mask_lofar = hp.ud_grade(mask_lofar, nside_out=args.nside)

    # Map
    npix = hp.nside2npix(args.nside)
    ipix = hp.ang2pix(args.nside, cat['RA'], cat['DEC'], lonlat=True)
    map_n = np.bincount(ipix, minlength=npix).astype(float)
    good_pix = mask_lofar > 0.
    mean_n = np.sum(map_n[good_pix])/np.sum(mask_lofar[good_pix])
    map_lofar = np.zeros(npix)
    map_lofar[good_pix] = map_n[good_pix]/(mask_lofar[good_pix]*mean_n)-1

    if args.verbose:
        print('----> Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()


# Power spectra
if args.verbose:
    print('Calculating power spectra')
    sys.stdout.flush()

# 1. Define fields
if args.verbose:
    print('----> Fields. ', end='')  # noqa: E999
    sys.stdout.flush()
time_start = time.time()
b = nmt.NmtBin(args.nside, nlb=50)
if args.run_planck:
    f_p = nmt.NmtField(mask_planck, [map_planck], n_iter=0)
if args.run_lofar:
    f_l = nmt.NmtField(mask_lofar, [map_lofar], n_iter=0)
if args.verbose:
    print('Done in {:.2f} secs'.format(time.time()-time_start))
    sys.stdout.flush()

# 2. Compute mode coupling matrix (Planck-Planck)
if args.run_planck:
    if args.verbose:
        print('----> MCM (Planck-Planck). ', end='')  # noqa: E999
        sys.stdout.flush()
    time_start = time.time()
    w_pp = nmt.NmtWorkspace()
    if os.path.isfile('outputs/mcm_pp.fits') and not args.recompute_mcm:
        w_pp.read_from('outputs/mcm_pp.fits')
    else:
        w_pp.compute_coupling_matrix(f_p, f_p, b)
        w_pp.write_to('outputs/mcm_pp.fits')
    if args.verbose:
        print('Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()

# 3. Compute mode coupling matrix (Planck-Lofar)
if args.run_planck and args.run_lofar:
    if args.verbose:
        print('----> MCM (Planck-Lofar). ', end='')  # noqa: E999
        sys.stdout.flush()
    time_start = time.time()
    w_pl = nmt.NmtWorkspace()
    if os.path.isfile('outputs/mcm_pl.fits') and not args.recompute_mcm:
        w_pl.read_from('outputs/mcm_pl.fits')
    else:
        w_pl.compute_coupling_matrix(f_p, f_l, b)
        w_pl.write_to('outputs/mcm_pl.fits')
    if args.verbose:
        print('Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()

# 4. Compute mode coupling matrix (Lofar-Lofar)
if args.run_lofar:
    if args.verbose:
        print('----> MCM (Lofar-Lofar). ', end='')  # noqa: E999
        sys.stdout.flush()
    time_start = time.time()
    w_ll = nmt.NmtWorkspace()
    if os.path.isfile('outputs/mcm_ll.fits') and not args.recompute_mcm:
        w_ll.read_from('outputs/mcm_ll.fits')
    else:
        w_ll.compute_coupling_matrix(f_l, f_l, b)
        w_ll.write_to('outputs/mcm_ll.fits')
    if args.verbose:
        print('Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()

# 5. Compute Cls
if args.verbose:
    print('----> Cls. ', end='')  # noqa: E999
    sys.stdout.flush()
time_start = time.time()
l_eff = b.get_effective_ells()
if args.run_planck:
    cl_pp = w_pp.decouple_cell(nmt.compute_coupled_cell(f_p, f_p))[0]
if args.run_planck and args.run_lofar:
    cl_pl = w_pl.decouple_cell(nmt.compute_coupled_cell(f_p, f_l))[0]
if args.run_lofar:
    cl_ll = w_ll.decouple_cell(nmt.compute_coupled_cell(f_l, f_l))[0]
if args.verbose:
    print('Done in {:.2f} secs'.format(time.time()-time_start))
    sys.stdout.flush()


# Covariance matrix
if args.verbose:
    print('Calculating covariance matrices')
    sys.stdout.flush()

# 1. Theory power spectra
if args.verbose:
    print('----> Cls theory. ', end='')  # noqa: E999
    sys.stdout.flush()
time_start = time.time()
l_arr = np.arange(3 * args.nside)
if args.run_planck:
    cl_th_pp = np.zeros(len(l_arr))
    nl_pp = np.zeros(len(l_arr))
    cl_f = np.loadtxt(os.path.join(args.path_planck, 'nlkk.dat'), unpack=True)
    cl_th_pp[int(cl_f[0, 0]):int(cl_f[0, -1])+1] = cl_f[2]
    nl_pp[int(cl_f[0, 0]):int(cl_f[0, -1])+1] = cl_f[1]
if args.run_planck and args.run_lofar:
    nl_pl = np.zeros(len(l_arr))
    cl_th_pl = np.zeros(len(l_arr))
if args.run_lofar:
    # Shot noise
    n_dens = mean_n * npix / (4 * np.pi)
    nl_coupled = np.ones(3 * args.nside) * np.mean(mask_lofar) / n_dens
    nl_ll = w_ll.decouple_cell([nl_coupled])[0]
    # Theory power spectra
    alpha_fit = np.log((cl_ll-nl_ll)[-1] / (cl_ll-nl_ll)[0]) / \
        np.log(l_eff[-1] / l_eff[0])
    if np.isnan(alpha_fit):
        alpha_fit = -1.3
    cl_th_ll = (cl_ll-nl_ll)[len(l_eff) // 3] * \
        ((l_arr+10) / l_eff[len(l_eff) // 3])**alpha_fit + np.mean(nl_ll)
if args.verbose:
    print('Done in {:.2f} secs'.format(time.time()-time_start))
    sys.stdout.flush()

# 2. Compute coupling coefficients (Planck-Planck-Planck-Planck)
if args.run_planck:
    if args.verbose:
        print('----> CMCM (PPPP). ', end='')  # noqa: E999
        sys.stdout.flush()
    time_start = time.time()
    cw_pppp = nmt.NmtCovarianceWorkspace()
    if os.path.isfile('outputs/cmcm_pppp.fits') and not args.recompute_mcm:
        cw_pppp.read_from('outputs/cmcm_pppp.fits')
    else:
        cw_pppp.compute_coupling_coefficients(fla1=f_p, fla2=f_p,
                                              flb1=f_p, flb2=f_p)
        cw_pppp.write_to('outputs/cmcm_pppp.fits')
    if args.verbose:
        print('Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()

# 3. Compute coupling coefficients (Planck-Planck-Planck-Lofar)
if args.run_planck and args.run_lofar:
    if args.verbose:
        print('----> CMCM (PPPL). ', end='')  # noqa: E999
        sys.stdout.flush()
    time_start = time.time()
    cw_pppl = nmt.NmtCovarianceWorkspace()
    if os.path.isfile('outputs/cmcm_pppl.fits') and not args.recompute_mcm:
        cw_pppl.read_from('outputs/cmcm_pppl.fits')
    else:
        cw_pppl.compute_coupling_coefficients(fla1=f_p, fla2=f_p,
                                              flb1=f_p, flb2=f_l)
        cw_pppl.write_to('outputs/cmcm_pppl.fits')
    if args.verbose:
        print('Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()

# 4. Compute coupling coefficients (Planck-Planck-Lofar-Lofar)
if args.run_planck and args.run_lofar:
    if args.verbose:
        print('----> CMCM (PPLL). ', end='')  # noqa: E999
        sys.stdout.flush()
    time_start = time.time()
    cw_ppll = nmt.NmtCovarianceWorkspace()
    if os.path.isfile('outputs/cmcm_ppll.fits') and not args.recompute_mcm:
        cw_ppll.read_from('outputs/cmcm_ppll.fits')
    else:
        cw_ppll.compute_coupling_coefficients(fla1=f_p, fla2=f_p,
                                              flb1=f_l, flb2=f_l)
        cw_ppll.write_to('outputs/cmcm_ppll.fits')
    if args.verbose:
        print('Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()

# 5. Compute coupling coefficients (Planck-Lofar-Planck-Lofar)
if args.run_planck and args.run_lofar:
    if args.verbose:
        print('----> CMCM (PLPL). ', end='')  # noqa: E999
        sys.stdout.flush()
    time_start = time.time()
    cw_plpl = nmt.NmtCovarianceWorkspace()
    if os.path.isfile('outputs/cmcm_plpl.fits') and not args.recompute_mcm:
        cw_plpl.read_from('outputs/cmcm_plpl.fits')
    else:
        cw_plpl.compute_coupling_coefficients(fla1=f_p, fla2=f_l,
                                              flb1=f_p, flb2=f_l)
        cw_plpl.write_to('outputs/cmcm_plpl.fits')
    if args.verbose:
        print('Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()

# 6. Compute coupling coefficients (Planck-Lofar-Lofar-Lofar)
if args.run_planck and args.run_lofar:
    if args.verbose:
        print('----> CMCM (PLLL). ', end='')  # noqa: E999
        sys.stdout.flush()
    time_start = time.time()
    cw_plll = nmt.NmtCovarianceWorkspace()
    if os.path.isfile('outputs/cmcm_plll.fits') and not args.recompute_mcm:
        cw_plll.read_from('outputs/cmcm_plll.fits')
    else:
        cw_plll.compute_coupling_coefficients(fla1=f_p, fla2=f_l,
                                              flb1=f_l, flb2=f_l)
        cw_plll.write_to('outputs/cmcm_plll.fits')
    if args.verbose:
        print('Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()

# 7. Compute coupling coefficients (Lofar-Lofar-Lofar-Lofar)
if args.run_lofar:
    if args.verbose:
        print('----> CMCM (LLLL). ', end='')  # noqa: E999
        sys.stdout.flush()
    time_start = time.time()
    cw_llll = nmt.NmtCovarianceWorkspace()
    if os.path.isfile('outputs/cmcm_llll.fits') and not args.recompute_mcm:
        cw_llll.read_from('outputs/cmcm_llll.fits')
    else:
        cw_llll.compute_coupling_coefficients(fla1=f_l, fla2=f_l,
                                              flb1=f_l, flb2=f_l)
        cw_llll.write_to('outputs/cmcm_llll.fits')
    if args.verbose:
        print('Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()

# 8. Compute Covariance matrix
if args.verbose:
    print('----> Covariance matrix. ', end='')  # noqa: E999
    sys.stdout.flush()
time_start = time.time()
if args.run_planck:
    cov_pppp = nmt.gaussian_covariance(cw_pppp, 0, 0, 0, 0,
                                       [cl_th_pp], [cl_th_pp],
                                       [cl_th_pp], [cl_th_pp],
                                       wa=w_pp, wb=w_pp)
if args.run_planck and args.run_lofar:  # cla1b1, cla1b2, cla2b1, cla2b2
    cov_pppl = nmt.gaussian_covariance(cw_pppl, 0, 0, 0, 0,
                                       [cl_th_pp], [cl_th_pl],
                                       [cl_th_pp], [cl_th_pl],
                                       wa=w_pp, wb=w_pl)
    cov_ppll = nmt.gaussian_covariance(cw_ppll, 0, 0, 0, 0,
                                       [cl_th_pl], [cl_th_pl],
                                       [cl_th_pl], [cl_th_pl],
                                       wa=w_pp, wb=w_ll)
    cov_plpl = nmt.gaussian_covariance(cw_plpl, 0, 0, 0, 0,
                                       [cl_th_pp], [cl_th_pl],
                                       [cl_th_pl], [cl_th_ll],
                                       wa=w_pl, wb=w_pl)
    cov_plll = nmt.gaussian_covariance(cw_plll, 0, 0, 0, 0,
                                       [cl_th_pl], [cl_th_pl],
                                       [cl_th_ll], [cl_th_ll],
                                       wa=w_pl, wb=w_ll)
if args.run_lofar:
    cov_llll = nmt.gaussian_covariance(cw_llll, 0, 0, 0, 0,
                                       [cl_th_ll], [cl_th_ll],
                                       [cl_th_ll], [cl_th_ll],
                                       wa=w_ll, wb=w_ll)
if args.verbose:
    print('Done in {:.2f} secs'.format(time.time()-time_start))
    sys.stdout.flush()


# Write output
if args.run_planck and args.run_lofar:
    np.savez(args.output,
             l_eff=l_eff, cl_pp=cl_pp, cl_pl=cl_pl, cl_ll=cl_ll,
             l_arr=l_arr, nl_pp=nl_pp, nl_pl=nl_pl, nl_ll=nl_ll,
             cl_th_pp=cl_th_pp, cl_th_pl=cl_th_pl, cl_th_ll=cl_th_ll,
             cov_pppp=cov_pppp, cov_pppl=cov_pppl, cov_ppll=cov_ppll,
             cov_plpl=cov_plpl, cov_plll=cov_plll, cov_llll=cov_llll)


#print(l_eff.shape)
#print(l_arr.shape)
#print(cl_pp.shape)
#print(nl_pp.shape)
#print(cl_th_pp.shape)
#print(cov_pppp.shape)

# Plotting
if args.plot_stuff:

    # 1. Planck-Planck
    if args.run_planck:
        plt.figure()
        err_pppp = np.sqrt(np.diag(cov_pppp))
        plt.errorbar(l_eff, cl_pp - nl_pp, yerr=err_pppp, fmt='r.',
                     label=r'Data')
        plt.plot(l_arr, cl_th_pp - np.mean(nl_pp), 'k-', label='Planck signal')
        plt.plot(l_eff, nl_pp, 'g--', label='Planck noise')
        plt.loglog()
        plt.xlim([0.9*l_eff[0], 1.1*l_eff[-1]])
        plt.ylim([0.5*np.amin((cl_pp-nl_pp)-err_pppp),
                  2*np.amax((cl_pp-nl_pp)+err_pppp)])
        plt.xlabel(r'$\ell$', fontsize=14)
        plt.ylabel(r'$C_\ell^{PP}$', fontsize=14)
        plt.legend(loc='upper right')

    # 2. Planck-Lofar
    if args.run_planck and args.run_lofar:
        plt.figure()
        err_plpl = np.sqrt(np.diag(cov_plpl))
        plt.errorbar(l_eff, cl_pl - nl_pl, yerr=err_plpl, fmt='r.',
                     label=r'Data')
        plt.plot(l_arr, cl_th_pl - np.mean(nl_pl), 'k-', label='Zero signal')
        plt.plot(l_eff, nl_pl, 'g--', label='Zero noise')
        plt.loglog()
        plt.xlim([0.9*l_eff[0], 1.1*l_eff[-1]])
        plt.ylim([0.5*np.amin((cl_pl-nl_pl)-err_plpl),
                  2*np.amax((cl_pl-nl_pl)+err_plpl)])
        plt.xlabel(r'$\ell$', fontsize=14)
        plt.ylabel(r'$C_\ell^{PL}$', fontsize=14)
        plt.legend(loc='upper right')

    # 3. Lofar-Lofar
    if args.run_lofar:
        plt.figure()
        err_llll = np.sqrt(np.diag(cov_llll))
        plt.errorbar(l_eff, cl_ll - nl_ll, yerr=err_llll, fmt='r.',
                     label=r'Data, $I<%.1lf {\rm mJy}$' % args.I_thr)
        plt.plot(l_arr, cl_th_ll - np.mean(nl_ll), 'k-', label='Power-law fit')
        plt.plot(l_eff, nl_ll, 'g--', label='Shot noise prediction')
        plt.loglog()
        plt.xlim([0.9*l_eff[0], 1.1*l_eff[-1]])
        plt.ylim([0.5*np.amin((cl_ll-nl_ll)-err_llll),
                  2*np.amax((cl_ll-nl_ll)+err_llll)])
        plt.xlabel(r'$\ell$', fontsize=14)
        plt.ylabel(r'$C_\ell^{LL}$', fontsize=14)
        plt.legend(loc='upper right')

    # 4. Maps
    if args.run_planck:
        utils.plot_lotss_map(map_planck, title='Map Planck')
        utils.plot_lotss_map(mask_planck, title='Mask Planck')
    if args.run_lofar:
        utils.plot_lotss_map(map_lofar, title='Map Lofar')
        utils.plot_lotss_map(mask_lofar, title='Mask Lofar')
    plt.show()


if args.verbose:
    print('Success!')
