import utils as ut
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--nz-choice', type=str, default='w',
                    help='\'w\', \'s3\', or \'vc\'')
parser.add_argument('--bz-choice', type=str, default='inv_growth',
                    help='\'inv_growth\' or \'constant\'')
parser.add_argument('--data', type=str, default='gg-gk',
                    help='Data (separated by hyphens)')
parser.add_argument('--vary-nz', default=False, action='store_true',
                    help='Vary N(z) tail?')
parser.add_argument('--fix-bz', default=False, action='store_true',
                    help='Fix b(z)?')
parser.add_argument('--fix-s8', default=False, action='store_true',
                    help='Fix sigma8?')
parser.add_argument('--lmax', type=float, default=500,
                    help='l_max')
parser.add_argument('--n-samples', type=int, default=10000,
                    help='Number of samples')
parser.add_argument('--n-walkers', type=int, default=8,
                    help='Number of walkers')
parser.add_argument('--use-mpi', default=False, action='store_true',
                    help='Use MPI')
parser.add_argument('--prefix', type=str,
                    default='/mnt/extraspace/damonge/LotssCross/out_2048_pfull_hrmask_deproj/',
                    help='Input/output predir')
o = parser.parse_args()

pars, fname = ut.mk_sampler_params(o.prefix,
                                   nz_choice=o.nz_choice,
                                   bz_choice=o.bz_choice,
                                   nz_sample=o.vary_nz,
                                   bz_sample=not o.fix_bz,
                                   s8_sample=not o.fix_s8,
                                   data=o.data.split('-'),
                                   nsteps=o.n_samples,
                                   nwalkers=o.n_walkers,
                                   l_max=o.lmax)
ut.sample(pars, fname, plot_stuff=False, use_mpi=o.use_mpi)
