#!/bin/bash

pyex="addqueue -q cmb -m 0.5 -n 1 /usr/bin/python3"

# bz alone
for nz in w s3 vc
do
    for bz in inv_growth constant plateau
    do
	for data in gg gk gg-gk
	do
	    ${pyex} sampler.py --nz-choice ${nz} --bz-choice ${bz} --data ${data} --fix-s8 --n-samples 3000 --n-walkers 8 --prefix /mnt/extraspace/damonge/LotssCross/out_2048_pfull_hrmask_deproj/
	done
    done
done

# bz and nz
for nz in ana w s3 vc
do
    for bz in inv_growth constant plateau
    do
	${pyex} sampler.py --nz-choice ${nz} --bz-choice ${bz} --data gg-gk --vary-nz --fix-s8 --n-samples 10000 --n-walkers 8 --prefix /mnt/extraspace/damonge/LotssCross/out_2048_pfull_hrmask_deproj/
    done
done

pyex="addqueue -q cmb -n 24 /usr/bin/python3"

# bz and s8
for nz in w s3 vc
do
    for bz in inv_growth constant plateau
    do
	${pyex} sampler.py --nz-choice ${nz} --bz-choice ${bz} --data gg-gk --n-samples 10000 --n-walkers 24 --prefix /mnt/extraspace/damonge/LotssCross/out_2048_pfull_hrmask_deproj/ --use-mpi
    done
done

#  -h, --help            show this help message and exit
#  --nz-choice NZ_CHOICE
#                        'w', 's3', or 'vc'
#  --bz-choice BZ_CHOICE
#                        'inv_growth' or 'constant'
#  --data DATA           Data (separated by hyphens)
#  --vary-nz             Vary N(z) tail?
#  --fix-bz              Fix b(z)?
#  --fix-s8              Fix sigma8?
#  --lmax LMAX           l_max
#  --n-samples N_SAMPLES
#                        Number of samples
#  --n-walkers N_WALKERS
#                        Number of walkers
#  --use-mpi             Use MPI
#  --prefix PREFIX       Input/output predir
#addqueue -q cmb -m 0.5 -n 12 /usr/bin/python3 mcmc.py --param-file params_dam_tinker10.yml --use-mpi --data-name wisc5
