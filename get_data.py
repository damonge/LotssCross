from utils import Pointings, dwl_file, untar_file


print("\n** Downloading Planck lensing map")
dwl_file("http://pla.esac.esa.int/pla/aio/product-action?COSMOLOGY.FILE_ID=COM_Lensing_2048_R2.00.tar",
         'data/planck_lensing_compressed.tar')
untar_file('data/planck_lensing_compressed.tar', path='data/planck')

print("\n** Downloading Lofar catalogs")
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
pt = Pointings()
pt.download_pointings(pt.pointings['name'])
