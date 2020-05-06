import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib import rc
rc('font', **{'family': 'sans-serif',
              'sans-serif': ['Helvetica'],
              'size': 15})
rc('text', usetex=True)


def plot_comp(dirs, fname_out=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
    plt.subplots_adjust(hspace=0)
    cols = ['r', 'g', 'b', 'y', 'm', 'c']
    ndir = len(dirs)
    ds = [np.load(os.path.join(d['dirname'], 'cls.npz')) for d in dirs]
    size = np.amin([len(d['cl_gg']) for d in ds])
    for i, d in enumerate(ds[1:]):
        l = d['l_eff'][:size]
        dl = -6 + 12 * i /(ndir-1)
        c = cols[i % 7]
        dx = ((d['cl_gg']-d['nl_gg'])[:size] -
              (ds[0]['cl_gg']-ds[0]['nl_gg'])[:size])
        er = np.sqrt(np.diag(np.fabs(ds[0]['cov_gggg'])))[:size]
        ax1.errorbar(l + dl, dx / er, yerr=er/er,
                     fmt=c+'.', label=dirs[i+1]['name'])
        dx = (d['cl_gk'][:size]-ds[0]['cl_gk'][:size])
        er = np.sqrt(np.diag(np.fabs(ds[0]['cov_gkgk'])))[:size]
        ax2.errorbar(l + dl, dx / er, yerr=er/er,
                     fmt=c+'.', label=dirs[i+1]['name'])
        ax1.plot([10, 512], [0, 0], 'k--', lw=1)
        ax2.plot([10, 512], [0, 0], 'k--', lw=1)
    ax2.set_xlabel(r'$\ell$', fontsize=16)
    ax1.set_ylabel(r'$\Delta C^{gg}_\ell / \sigma_\ell^{gg}$', fontsize=16)
    ax2.set_ylabel(r'$\Delta C^{g\kappa}_\ell / \sigma_\ell^{g\kappa}$', fontsize=16)
    for ax in [ax1, ax2]:
        ax.set_xlim([10, 512])
    ax1.legend(loc='upper right', frameon=False,
              ncol=2, labelspacing=0.1)

    if fname_out is not None:
        plt.savefig(fname_out, bbox_inches='tight')


plot_comp([{'name': 'Fiducial',
            'dirname': '/mnt/extraspace/damonge/LotssCross/out_2048_pfull_hrmask'},
           {'name': r'$N_{\rm side} = 256$',
            'dirname': '/mnt/extraspace/damonge/LotssCross/out_256_pfull'},
           {'name': r'${\rm Sys.\,\,deprojected}$',
            'dirname': '/mnt/extraspace/damonge/LotssCross/out_2048_pfull_hrmask_deproj'}
           ], fname_out='plots/cl_syst.pdf')
plt.show()
