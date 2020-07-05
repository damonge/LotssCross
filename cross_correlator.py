import os
import sys
import time
import argparse
import numpy as np
import healpy as hp
import pymaster as nmt
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.integrate import simps
import matplotlib.pyplot as plt
import utils as ut
import pyccl as ccl


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
parser.add_argument('--mask-planck-extra', default=False, action='store_true',
                    help='Mask Planck on LoFAR footprint? (default: False)')
parser.add_argument('--cut_peak', default=False, action='store_true',
                    help='Use peak flux? (default: False)')
parser.add_argument('--I_thr', default=2.,  type=float,
                    help='Flux threshold (default: 2 mJy)')
parser.add_argument('--use_median', default=False, action='store_true',
                    help='Use median rms (instead of mean)')
parser.add_argument('--use-hires-mask', default=False, action='store_true',
                    help='Use high-res pointing-based mask')
parser.add_argument('--recompute_mcm', default=False, action='store_true',
                    help='Recompute MCM? (default: False)')
parser.add_argument('--plot-stuff', default=False, action='store_true',
                    help='Make plots? (default: False)')
parser.add_argument('--deproject-ivar', default=False, action='store_true',
                    help='Deproject inverse-variance map?')
parser.add_argument('--silent', '-s', dest='verbose', default=True,
                    action='store_false', help='Verbose mode (default: False)')
parser.add_argument('--just-save-nz', default=False, action='store_true',
                    help='Quit after saving N(z) (default: False)')
parser.add_argument('--output-dir', '-o', type=str, default='output_cls_cov',
                    help='Output directory, default=output_cls_cov')
args = parser.parse_args()

os.system('mkdir -p ' + args.output_dir)


fields = []
id0 = 0
field_ids = {}

# Lofar maps
if args.run_lofar:
    if args.verbose:
        print('Creating Lofar maps')
        sys.stdout.flush()
    time_start = time.time()

    # Read catalog
    fname_rc = os.path.join(args.path_lofar, 'radio_catalog.fits')
    fname_vac = os.path.join(args.path_lofar, 'hetdex_optical_ids.fits')
    cat_rc = fits.open(fname_rc)[1].data
    cat_vac = fits.open(fname_vac)[1].data

    if args.use_rc:
        cat = cat_rc
    else:
        cat = cat_vac

    # Flux cut
    if args.cut_peak:
        flux_flag = 'Peak_flux'
    else:
        flux_flag = 'Total_flux'
    cat = cat[cat[flux_flag] >= args.I_thr]
    cat_rc = cat_rc[cat_rc[flux_flag] >= args.I_thr]
    cat_vac = cat_vac[cat_vac[flux_flag] >= args.I_thr]

    # p-map
    if args.use_median:
        meanmed = 'median'
    else:
        meanmed = 'mean'
    fname_p = "outputs/p_map_rms_" + meanmed + ("_Imin%.3lf.fits" % args.I_thr)
    p_map = hp.read_map(fname_p, dtype=None, verbose=False).astype(float)
    p_map /= np.amax(p_map)
    p_map = hp.ud_grade(p_map, nside_out=args.nside)
    # Footprint mask
    if args.use_hires_mask:
        msk_b = hp.read_map("outputs/pointings_hp2048_mask_good.fits.gz",
                            dtype=None, verbose=False).astype(float)
    else:
        msk_b = hp.read_map("outputs/mask_d_256.fits", dtype=None,
                            verbose=False).astype(float)
    msk_b = hp.ud_grade(msk_b, nside_out=args.nside)

    # Mask
    mask_lofar = p_map * msk_b
    mask_lofar[mask_lofar < 0.5] = 0

    # Deprojection?
    if args.deproject_ivar:
        temp_deproj = hp.read_map("outputs/pointings_hp2048_ivar_good.fits.gz",
                                  dtype=None, verbose=False).astype(float)
        temp_deproj = hp.ud_grade(temp_deproj, nside_out=args.nside)
        temp_mean = np.sum(temp_deproj * mask_lofar) / np.sum(mask_lofar)
        temp_deproj = temp_deproj / temp_mean - 1
        temp_deproj[mask_lofar <= 0.01] = 0
    else:
        temp_deproj = None

    # Map
    npix = hp.nside2npix(args.nside)
    ipix = hp.ang2pix(args.nside, cat['RA'], cat['DEC'], lonlat=True)
    map_n = np.bincount(ipix, minlength=npix).astype(float)

    # N(z)
    ipix_vac = hp.ang2pix(256, cat_vac['RA'], cat_vac['DEC'], lonlat=True)
    dat = cat_vac[mask_lofar[ipix] > 0.]

    # First, calculate flux-based weights
    id_wz = ~np.isnan(dat['z_best'])
    li_max = np.log10(np.amax(dat['Total_flux']))
    n_all, li_bins = np.histogram(np.log10(dat['Total_flux']), bins=20, range=[0, li_max])
    li = 0.5 * (li_bins[1:] + li_bins[:-1])
    n_wz, _ = np.histogram(np.log10(dat['Total_flux'][id_wz]), bins=20, range=[0, li_max])
    w = np.ones(20)
    w[n_wz>0] = n_all[n_wz>0] * np.sum(id_wz) / (n_wz[n_wz>0] * len(id_wz))
    wfunc = interp1d(li, w, bounds_error=False, fill_value=(w[0], w[-1]))
    weights = wfunc(np.log10(dat['Total_flux']))

    # Then, compute weighted and unweighted redshift distributions
    def get_nz(z_arr, w_arr=None):
        nz, z = np.histogram(z_arr, bins=100,
                             range=[0, 10], density=True,
                             weights=w_arr)
        z = 0.5 * (z[1:] + z[:-1])
        nzb = np.zeros(len(nz)+1)
        nzb[1:] = nz
        nz = nzb
        zb = np.zeros(len(z)+1)
        zb[1:] = z
        z = zb
        return z, nz
    z, nz = get_nz(dat['z_best'][id_wz])
    _, nz_w = get_nz(dat['z_best'][id_wz], weights[id_wz])
    csm = ut.get_default_cosmo()
    bz = 2.0/ccl.growth_factor(csm, 1./(1+z))

    nz_dict = {'z_g': z,
               'nz_g': nz,
               'nz_g_w': nz_w,
               'bz_g': bz}
    # Combine with SKADS N(z)s
    try:
        d_skads = np.load("data/nz_skads_flux%.3lf.npz" % args.I_thr)
        z_sk = d_skads['zs']
        nz_sk = d_skads['nz']
        if not np.all(z_sk == z):
            raise ValueError("Redshifts are not aligned. Exiting")
        norm_skads = simps(nz_sk, x=z_sk)
        nz_sk /= norm_skads
        norm_lotss = simps(nz_w, x=z)
        nz_w /= norm_lotss
        z_cut = 0.5
        F_skads_zoverzcut = simps(nz_sk[z_sk>z_cut], x=z_sk[z_sk>z_cut])
        F_lotss_zoverzcut = simps(nz_w[z>z_cut], x=z[z>z_cut])
        nz_comb = np.zeros_like(nz)
        nz_comb[z<=z_cut] = nz_w[z<=z_cut]
        nz_comb[z>z_cut] = nz_sk[z>z_cut] * F_lotss_zoverzcut / F_skads_zoverzcut
        nz_dict['nz_g_s3'] = nz_sk
        nz_dict['nz_g_s3_comb'] = nz_comb
    except FileNotFoundError:
        raise ValueError("SKADs N(z) not found, won't combine N(z)s")

    # Combine with VLACOS N(z)s
    try:
        d_vlacos = np.load("data/nz_vlacos_flux%.3lf.npz" % args.I_thr)
        z_vc = d_vlacos['zs']
        nz_vc = d_vlacos['nz']
        norm_vlacos = simps(nz_vc, x=z_vc)
        nz_vc /= norm_vlacos
        nz_dict['z_g_vc'] = z_vc
        nz_dict['nz_g_vc'] = nz_vc
    except FileNotFoundError:
        print("VLA-COSMOS N(z) not found, won't combine N(z)s")

    np.savez(os.path.join(args.output_dir, 'nz'), **nz_dict)
    if args.just_save_nz:
        print("Saved N(z), exiting")
        exit(1)

    fields.append(ut.Field('lofar_g', 'g', map_n, mask_lofar,
                           nz=(z, nz_sk), bz=bz, templates=temp_deproj))
    field_ids['g'] = id0
    id0 += 1
    if args.verbose:
        print('----> Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()

# Planck maps
if args.run_planck:
    if args.verbose:
        print('Creating Planck maps')
        sys.stdout.flush()
    time_start = time.time()

    # Load files
    mask_planck = hp.read_map(os.path.join(args.path_planck, 'mask_ns%d.fits' % (args.nside)),
                              dtype=None, verbose=False).astype(float)
    map_planck = hp.read_map(os.path.join(args.path_planck, 'map_kappa_ns%d.fits' % (args.nside)),
                                          dtype=None, verbose=False).astype(float)

    if args.mask_planck_extra:
        # Footprint mask
        if args.use_hires_mask:
            msk_b = hp.read_map("outputs/pointings_hp2048_mask_good.fits.gz",
                                dtype=None, verbose=False).astype(float)
        else:
            msk_b = hp.read_map("outputs/mask_d_256.fits", dtype=None,
                                verbose=False).astype(float)
        ns_planck = hp.get_nside(mask_planck)
        msk_b = hp.ud_grade(msk_b, nside_out=ns_planck)
        mask_planck[msk_b < 0.1] = 0
        map_planck[msk_b < 0.1] = 0

    # Upgrade maps
    mask_planck = hp.ud_grade(mask_planck, nside_out=args.nside)
    map_planck = hp.ud_grade(map_planck, nside_out=args.nside)

    fields.append(ut.Field('planck_k', 'k', map_planck, mask_planck))
    field_ids['k'] = id0
    id0 += 1
    if args.verbose:
        print('----> Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()
n_fields = len(fields)


# Define field iterator
def iterate_fields(unique=True):
    for i1 in range(n_fields):
        f1 = fields[i1]
        if unique:
            i20 = i1
        else:
            i20 = 0
        for i2 in range(i20, n_fields):
            f2 = fields[i2]
            pair = f1.kind + f2.kind
            yield(i1, f1, i2, f2, pair)


pair_names = [p for _, _, _, _, p in iterate_fields()]
ppair_names = []
for i1, p1 in enumerate(pair_names):
    for p2 in pair_names[i1:]:
        ppair_names.append(p1+p2)


def get_pair_name(f1, f2):
    p = f1.kind + f2.kind
    if p in pair_names:
        return p
    if p[::-1] in pair_names:
        return p[::-1]
    raise ValueError(f"{p} is not a valid pair name")


def get_ppair_name(pa, pb):
    if pa+pb in ppair_names:
        return pa+pb
    if pb+pa in ppair_names:
        return pb+pa
    raise ValueError(f"Couldn't find ppair {pa}{pb}")


l_arr = np.arange(3 * args.nside)
b = nmt.NmtBin(args.nside, nlb=50)
l_eff = b.get_effective_ells()


# Power spectra
if args.verbose:
    print('Calculating power spectra')
    sys.stdout.flush()

# 1. Compute mode coupling matrices
wsps = {}
for i1, f1, i2, f2, pair in iterate_fields():
    if args.verbose:
        print(f'----> MCM ({pair}). ', end='')  # noqa: E999
        sys.stdout.flush()
    time_start = time.time()
    fname = os.path.join(args.output_dir, f'mcm_{pair}.fits')
    w = nmt.NmtWorkspace()
    if os.path.isfile(fname) and not args.recompute_mcm:
        print("Reading...", end="")
        w.read_from(fname)
    else:
        w.compute_coupling_matrix(f1.f, f2.f, b, n_iter=0)
        w.write_to(fname)
    wsps[pair] = w
    if args.verbose:
        print('Done in {:.2f} secs'.format(time.time()-time_start))
        sys.stdout.flush()

# 2. Compute Cls
if args.verbose:
    print('----> Cls. ', end='')  # noqa: E999
    sys.stdout.flush()
time_start = time.time()
cls = {}
for i1, f1, i2, f2, pair in iterate_fields():
    cl_coupled = nmt.compute_coupled_cell(f1.f, f2.f)
    cls[pair] = wsps[pair].decouple_cell(cl_coupled)[0]
if args.verbose:
    print('Done in {:.2f} secs'.format(time.time()-time_start))
    sys.stdout.flush()

# Noise spectra
nls = {}
if args.run_lofar:
    nl_coupled = fields[field_ids['g']].nl_coupled
    nls['gg'] = wsps['gg'].decouple_cell([nl_coupled])[0]
    if args.run_planck:
        nls['gk'] = np.zeros_like(l_eff)
if args.run_planck:
    cl_f = np.loadtxt(os.path.join(args.path_planck, 'nlkk.dat'), unpack=True)
    nl_unbinned = np.zeros(len(l_arr))
    lmax = min(3*args.nside-1, int(cl_f[0, -1]))
    nl_unbinned[int(cl_f[0, 0]):lmax+1] = cl_f[1][cl_f[0] <= lmax]
    w = wsps['kk']
    nls['kk'] = w.decouple_cell(w.couple_cell([nl_unbinned]))[0]


# Theory spectra
cosmo = ut.get_default_cosmo()
cls_th = {}
if args.run_lofar:
    tg = fields[field_ids['g']].t
    clgg = ccl.angular_cl(cosmo, tg, tg, l_arr)
    clgg += np.mean(nls['gg'])
    cls_th['gg'] = clgg
    if args.run_planck:
        tk = fields[field_ids['k']].t
        clgk = ccl.angular_cl(cosmo, tg, tk, l_arr)
        cls_th['gk'] = clgk
if args.run_planck:
    cl_f = np.loadtxt(os.path.join(args.path_planck, 'nlkk.dat'), unpack=True)
    cl = np.zeros(len(l_arr))
    lmax = min(3*args.nside-1, int(cl_f[0, -1]))
    cl[int(cl_f[0, 0]):lmax+1] = cl_f[2][cl_f[0] <= lmax]
    cls_th['kk'] = cl


# Covariance matrix
if args.verbose:
    print('Calculating covariance matrices')
    sys.stdout.flush()

covs = {}
for ia1, fa1, ia2, fa2, pa in iterate_fields():
    for ib1, fb1, ib2, fb2, pb in iterate_fields():
        ppair = get_ppair_name(pa, pb)
        if args.verbose:
            print(f'----> cov ({ppair}). ', end='')  # noqa: E999
            sys.stdout.flush()
        time_start = time.time()
        fname = os.path.join(args.output_dir,
                             f'cmcm_{ppair}.fits')
        cw = nmt.NmtCovarianceWorkspace()
        if os.path.isfile(fname) and not args.recompute_mcm:
            print("Reading...", end="")
            cw.read_from(fname)
        else:
            cw.compute_coupling_coefficients(fla1=fa1.f, fla2=fa2.f,
                                             flb1=fb1.f, flb2=fb2.f,
                                             n_iter=0)
            cw.write_to(fname)

        cl_a1b1 = cls_th[get_pair_name(fa1, fb1)]
        cl_a1b2 = cls_th[get_pair_name(fa1, fb2)]
        cl_a2b1 = cls_th[get_pair_name(fa2, fb1)]
        cl_a2b2 = cls_th[get_pair_name(fa2, fb2)]
        covs[ppair] = nmt.gaussian_covariance(cw, 0, 0, 0, 0,
                                              [cl_a1b1], [cl_a1b2],
                                              [cl_a2b1], [cl_a2b2],
                                              wsps[pa], wsps[pb])
        if args.verbose:
            print('Done in {:.2f} secs'.format(time.time()-time_start))
            sys.stdout.flush()

dsave = {}
dsave['l_eff'] = l_eff
for p in pair_names:
    dsave['cl_' + p] = cls[p]
    dsave['nl_' + p] = nls[p]
    dsave['cl_th_' + p] = cls_th[p]
for pp in ppair_names:
    dsave['cov_' + pp] = covs[pp]
dsave['z_g'] = z
dsave['nz_g'] = nz
dsave['bz_g'] = bz
np.savez(os.path.join(args.output_dir, 'cls'), **dsave)

# Plotting
if args.plot_stuff:
    for p in pair_names:
        plt.figure()
        cl = cls[p]
        nl = nls[p]
        clth = cls_th[p]
        pp = get_ppair_name(p, p)
        err = np.sqrt(np.diag(covs[pp]))
        plt.errorbar(l_eff, cl, yerr=err, fmt='r.',
                     label=r'Data')
        plt.plot(l_arr, clth, 'k-', label='Cov. model')
        if not np.all(nl == 0):
            plt.plot(l_eff, nl, 'g--', label='Noise bias')
        plt.loglog()
        plt.xlim([0.9*l_eff[0], 1.1*l_eff[-1]])
        plt.xlabel(r'$\ell$', fontsize=14)
        plt.ylabel(r'$C_\ell^{%s}$' % p, fontsize=14)
        plt.legend(loc='upper right')
        fname = os.path.join(args.output_dir, f'cl_{p}.png')
        plt.savefig(fname, bbox_inches='tight')

    for f in fields:
        n = f.name
        k = f.kind
        ut.plot_lotss_map(f.mp, title=f'Map {n}')
        plt.savefig(os.path.join(args.output_dir,
                                 f'map_{k}.png'),
                    bbox_inches='tight')
        ut.plot_lotss_map(f.msk, title=f'Mask {n}')
        plt.savefig(os.path.join(args.output_dir,
                                 f'mask_{k}.png'),
                    bbox_inches='tight')
    plt.show()


if args.verbose:
    print('Success!')
