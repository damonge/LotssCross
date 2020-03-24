import healpy as hp
from utils import FluxPDF, plot_lotss_map
import matplotlib.pyplot as plt

fname_out = "outputs/p_map_rms_median_Imin0p0.fits"
fname_rms = "outputs/map_rms_median_256.fits"
I_thr = 0.
snr_thr = 5

# Read mask and rms map
map_rms = hp.read_map(fname_rms, verbose=False)

# Read and initialize PDF
fpdf = FluxPDF()

# Compute probability map
p_map = fpdf.compute_p_map(snr_thr, map_rms, I_thr)
hp.write_map(fname_out, p_map, overwrite=True)

plot_lotss_map(p_map)
plt.show()
