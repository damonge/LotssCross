#!/bin/bash

#python_exec="addqueue -s -q cmb -m 1 -n 1x12 /usr/bin/python3"
python_exec="python3"
#python_exec="echo python3"
#arg_extra="--just-save-nz"
#arg_extra="--plot-stuff"

# I=2
#${python_exec} cross_correlator.py --nside 2048 -o /mnt/extraspace/damonge/LotssCross/out_2048_pfull ${arg_extra}
#${python_exec} cross_correlator.py --nside 2048 -o /mnt/extraspace/damonge/LotssCross/out_2048_pcut --mask-planck-extra ${arg_extra}
#${python_exec} cross_correlator.py --nside 2048 --use-hires-mask -o /mnt/extraspace/damonge/LotssCross/out_2048_pfull_hrmask ${arg_extra}
#${python_exec} cross_correlator.py --nside 2048 --use-hires-mask --mask-planck-extra -o /mnt/extraspace/damonge/LotssCross/out_2048_pcut_hrmask ${arg_extra}
#${python_exec} cross_correlator.py --nside 2048 --use-hires-mask --deproject-ivar -o /mnt/extraspace/damonge/LotssCross/out_2048_pfull_hrmask_deproj ${arg_extra}
#${python_exec} cross_correlator.py --nside 2048 --use-hires-mask --mask-planck-extra --deproject-ivar -o /mnt/extraspace/damonge/LotssCross/out_2048_pcut_hrmask_deproj ${arg_extra}
#${python_exec} cross_correlator.py --nside 256 -o /mnt/extraspace/damonge/LotssCross/out_256_pfull ${arg_extra}
#${python_exec} cross_correlator.py --nside 256 -o /mnt/extraspace/damonge/LotssCross/out_256_pcut --mask-planck-extra ${arg_extra}
${python_exec} cross_correlator.py --nside 256 --use-hires-mask -o /mnt/extraspace/damonge/LotssCross/out_256_pfull_hrmask ${arg_extra}
#${python_exec} cross_correlator.py --nside 256 --use-hires-mask --mask-planck-extra -o /mnt/extraspace/damonge/LotssCross/out_256_pcut_hrmask ${arg_extra}

# I=1
#${python_exec} cross_correlator.py --nside 2048 --I_thr 1 -o /mnt/extraspace/damonge/LotssCross/out_I1_2048_pfull ${arg_extra}
#${python_exec} cross_correlator.py --nside 2048 --I_thr 1 -o /mnt/extraspace/damonge/LotssCross/out_I1_2048_pcut --mask-planck-extra ${arg_extra}
#${python_exec} cross_correlator.py --nside 2048 --use-hires-mask --I_thr 1 -o /mnt/extraspace/damonge/LotssCross/out_I1_2048_pfull_hrmask ${arg_extra}
#${python_exec} cross_correlator.py --nside 2048 --use-hires-mask --mask-planck-extra --I_thr 1 -o /mnt/extraspace/damonge/LotssCross/out_I1_2048_pcut_hrmask ${arg_extra}
#${python_exec} cross_correlator.py --nside 2048 --use-hires-mask --I_thr 1 --deproject-ivar -o /mnt/extraspace/damonge/LotssCross/out_I1_2048_pfull_hrmask ${arg_extra}
#${python_exec} cross_correlator.py --nside 2048 --use-hires-mask --mask-planck-extra --I_thr 1 --deproject-ivar -o /mnt/extraspace/damonge/LotssCross/out_I1_2048_pcut_hrmask ${arg_extra}
#${python_exec} cross_correlator.py --nside 256 --I_thr 1 -o /mnt/extraspace/damonge/LotssCross/out_I1_256_pfull ${arg_extra}
#${python_exec} cross_correlator.py --nside 256 --I_thr 1 -o /mnt/extraspace/damonge/LotssCross/out_I1_256_pcut --mask-planck-extra ${arg_extra}
#${python_exec} cross_correlator.py --nside 256 --use-hires-mask --I_thr 1 -o /mnt/extraspace/damonge/LotssCross/out_I1_256_pfull_hrmask ${arg_extra}
#${python_exec} cross_correlator.py --nside 256 --use-hires-mask --mask-planck-extra --I_thr 1 -o /mnt/extraspace/damonge/LotssCross/out_I1_256_pcut_hrmask ${arg_extra}
