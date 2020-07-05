import numpy as np
import os
from astropy.io import fits
from utils import rotate_alm_g_c, rotate_map_g_c
import sys
import healpy as hp


if len(sys.argv)<2:
    print("Usage: planck_rotate.py nside")
    exit(1)

# Angular resolution
nside = int(sys.argv[1])
# Root data directory
predir_in = "data/planck/data/"
# Original alms
fname_alm = "dat_klm.fits"
# Mask
fname_msk = "mask.fits.gz"
# Output prefix
predir_out = "data/planck/data/"
os.system("mkdir -p " + predir_out)

# Read original alms
print("Reading kappa alms")
alm_g = hp.read_alm(predir_in + fname_alm)
# Rotate alms to Equatorial coordinates
print("Rotating to Equatorial")
alm_c = rotate_alm_g_c(alm_g)
# Compute map
print("Transforming to pixels")
mpk = hp.alm2map(alm_c, nside)
print("Writing output map")
hp.write_map(predir_out + "map_kappa_ns%d.fits" % nside,
             mpk, overwrite=True)

# Same thing with mask
print("Reading original mask")
msk_g = hp.read_map(predir_in + fname_msk, verbose=False)
print("Rotating to Equatorial")
msk_c = rotate_map_g_c(msk_g)
# Binarize
msk_c[msk_c < 0.5] = 0
msk_c[msk_c >= 0.5] = 1.
# Up/down-grade to chosen pixelization
msk_c = hp.ud_grade(msk_c, nside_out=nside)
print("Writing output mask")
hp.write_map(predir_out + "mask_ns%d.fits" % nside,
msk_c, overwrite=True)
