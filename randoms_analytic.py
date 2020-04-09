import healpy as hp
from utils import FluxPDF, plot_lotss_map
import matplotlib.pyplot as plt
from optparse import OptionParser


parser = OptionParser()
parser.add_option('--nside', dest='nside', default=256, type=int,
                  help='Nside parameter, default=256')
parser.add_option('--use-median', dest='use_median', default=False,
                  action='store_true',
                  help='Use median rms (instead of mean)')
parser.add_option('--Ithr', dest='I_thr', default=2.,  type=float,
                  help='Flux threshold (default: 2)')
parser.add_option('--snr', dest='snr', default=5,  type=int,
                  help='S/N threshold (default: 5)')
(o, args) = parser.parse_args()

if o.use_median:
    meanmed = 'median'
else:
    meanmed = 'mean'

fname_out = "outputs/p_map_rms_" + meanmed + ("_Imin%.3lf.fits" % o.I_thr)
fname_rms = "outputs/map_rms_" + meanmed + "_%d.fits" % (o.nside)

# Read mask and rms map
map_rms = hp.read_map(fname_rms, verbose=False)

# Read and initialize PDF
fpdf = FluxPDF()

# Compute probability map
p_map = fpdf.compute_p_map(o.snr, map_rms, o.I_thr)
hp.write_map(fname_out, p_map, overwrite=True)

plot_lotss_map(p_map)
plt.show()
