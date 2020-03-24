select floor(10*itot_1400)*0.1 as log_flux, count(*) as num from Galaxies group by floor(10*itot_1400)*0.1 order by log_flux
