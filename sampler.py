import utils as ut

nz_choice='w'
bz_choice='inv_growth'
data=['gk']
pars, fname = ut.mk_sampler_params('output_cls_cov/out_2048_pfull_hrmask_deproj/',
                                   nz_choice=nz_choice, bz_choice=bz_choice,
                                   nz_sample=False, bz_sample=True, s8_sample=False,
                                   data=data, nsteps=3000, nwalkers=6, l_max=500)
ut.sample(pars, fname, plot_stuff=False)
exit(1)
for nz_choice in ['w', 's3', 'vc']:
    for bz_choice in ['inv_growth', 'constant']:
        for data in [['gg'], ['gk'], ['gg', 'gk']]:
            print(nz_choice, bz_choice, data)
            pars, fname = ut.mk_sampler_params('output_cls_cov/out_2048_pfull_hrmask_deproj/',
                                               nz_choice=nz_choice, bz_choice=bz_choice,
                                               nz_sample=False, bz_sample=True, s8_sample=False,
                                               data=data, nsteps=3000, nwalkers=6, l_max=500)
            ut.sample(pars, fname, plot_stuff=False)


for nz_choice in ['ana', 'w', 's3', 'vc']:
    for bz_choice in ['inv_growth', 'constant']:
        data = ['gg', 'gk']
        print(nz_choice, bz_choice, data)
        pars, fname = ut.mk_sampler_params('output_cls_cov/out_2048_pfull_hrmask_deproj/',
                                           nz_choice=nz_choice, bz_choice=bz_choice,
                                           nz_sample=True, bz_sample=True, s8_sample=False,
                                           data=data, nsteps=10000, nwalkers=6, l_max=500)
        ut.sample(pars, fname, plot_stuff=True)
