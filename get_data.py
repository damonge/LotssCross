import numpy as np
from utils import get_pointing_names, dwl_file


print("\n** Downloading catalogs")
dwl_file("https://lofar-surveys.org/public/LOFAR_HBA_T1_DR1_catalog_v1.0.srl.fits",
         'data/radio_catalog.fits')
dwl_file("https://lofar-surveys.org/public/LOFAR_HBA_T1_DR1_merge_ID_optical_f_v1.2b_restframe.fits",
         'data/hetdex_optical_ids.fits')
dwl_file("https://lofar-surveys.org/public/LOFAR_HBA_T1_DR1_merge_ID_v1.2.comp.fits",
         'data/hetdex_optical_components.fits')
dwl_file("https://lofar-surveys.org/public/CatalogDescription.pdf",
         'data/description.pdf')
dwl_file("https://lofar-surveys.org/public/Mingo19_LoMorph_Cat.fits",
         'data/morph.fits')
dwl_file("https://lofar-surveys.org/public/mingo19_readme.txt",
         'data/readme_morph.txt')

print("\n**Downloading pointings")
for n in get_pointing_names():
    url_pre = "https://lofar-surveys.org/public/all-residual-mosaics/"
    url = url_pre + n + "-low-mosaic.fits"
    print(f"  {n}")
    dwl_file(url, 'data/res_low_' + n + '.fits', verbose=False)
