import numpy as np
import matplotlib.pyplot as plt
import theory as th
import os
import yaml


fname_pars = 'params.yml'
with open(fname_pars) as f:
    pyml = yaml.safe_load(f)

like = th.Like(pyml)

cl_d = like.get_data(split=True)
cov = like.get_covar(split=True)

# Best-fit and Fisher matrix
cl_t_ini = like.get_theory(like.p0, split=True)
if os.path.isfile(like.prefix_out+'_p0.npzzzz'):
    d = np.load(like.prefix_out+'_p0.npz')
    like.update_p0(d['p0'])
    fisher = d['fisher']
else:
    like.get_best_fit(update_p0=True)
    fisher = like.get_Fisher()
    np.savez(like.prefix_out+'_p0.npz',
             p0=like.p0, fisher=fisher)
print(like.p0,np.sqrt(np.diag(np.linalg.inv(fisher))))
cl_t_end = like.get_theory(like.p0, split=True)

# MCMC
like.sample()

# Plotting
like.get_chain()
plt.figure()
for p in like.chain[::10]:
    par = like.build_kwargs(p)
    z, nz = like.model.get_nz(**par)
    plt.plot(z, nz/np.amax(nz),'r-', alpha=0.05)
d = np.load('output_cls_cov/out_2048_pfull_hrmask_deproj/nz.npz')
plt.plot(d['z_g'], d['nz_g_w']/np.amax(d['nz_g_w']), 'k-', label='LoTSS')
plt.plot(d['z_g'], d['nz_g_s3']/np.amax(d['nz_g_s3']), 'k--', label='SKADS')
plt.plot(d['z_g_vc'], d['nz_g_vc']/np.amax(d['nz_g_vc']), 'k-.', label='VLA-COSMOS')
plt.xlim([0, 5])
plt.legend(loc='lower left', fontsize=15)
plt.xlabel(r'$z$', fontsize=15)
plt.ylabel(r'$N(z)$', fontsize=15)
plt.yscale('log')
plt.savefig(like.prefix_out+'nzs.pdf',
            bbox_inches='tight')

for i, (d, t_i, t_f) in enumerate(zip(cl_d, cl_t_ini, cl_t_end)):
    err = np.sqrt(np.fabs(np.diag(cov[i, :, i, :])))
    plt.figure()
    plt.title(like.cl_list[i])
    plt.errorbar(like.l, like.l*d, yerr=like.l*err, fmt='k.', label='Data')
    plt.plot(like.l, like.l*t_i, 'r-', label='Guess')
    plt.plot(like.l, like.l*t_f, 'b-', label='Best-fit')
    plt.plot(like.l, 0*like.l, 'k--')
    plt.xlabel(r'$\ell$', fontsize=16)
    plt.ylabel(r'$\ell\,C_\ell$', fontsize=16)
    plt.savefig(like.prefix_out+'cl_'+like.cl_list[i]+'.pdf',
                bbox_inches='tight')
print(like.p0, np.sqrt(np.diag(np.linalg.inv(fisher))),
      like.chi2(like.p0), like.chi2(like.p0)/like.nd_tot)

plt.show()
like.plot_chain(like.chain, save_figure=True)
