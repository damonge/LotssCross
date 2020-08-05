import numpy as np
import matplotlib.pyplot as plt
import pyccl as ccl
import warnings
import os


class DumPool(object):
    def __init__(self):
        pass

    def is_master(self):
        return True

    def close(self):
        pass

class SampleFileUtil(object):
    def __init__(self, filePrefix, rerun=False):
        self.filePrefix = filePrefix
        if rerun:
            mode = 'w'
        else:
            mode = 'a'
        self.samplesFile = open(self.filePrefix + '.txt', mode)
        self.probFile = open(self.filePrefix + 'prob.txt', mode)

    def persistSamplingValues(self, pos, prob):
        self.persistValues(self.samplesFile, self.probFile, pos, prob)

    def persistValues(self, posFile, probFile, pos, prob):
        posFile.write("\n".join(["\t".join([str(q) for q in p]) for p in pos]))
        posFile.write("\n")
        posFile.flush()

        probFile.write("\n".join([str(p) for p in prob]))
        probFile.write("\n")
        probFile.flush();

    def close(self):
        self.samplesFile.close()
        self.probFile.close()

    def __str__(self, *args, **kwargs):
        return "SampleFileUtil"
                                                                                                    
                                                                                
class Model(object):
    def __init__(self, nz_source, ell, z_name='z_g', nz_name='nz_g_w',
                 bias_type='constant'):
        self.sigma8 = -1.
        self._get_cosmo(0.81)
        self.bias_type = bias_type

        # N(z)
        if nz_source == 'analytic':
            self.nz_analytic = True
            self.z = np.linspace(0, 6, 512)
        else:
            self.nz_analytic = False
            nzd = np.load(nz_source)
            self.z = nzd[z_name]
            self.nz = nzd[nz_name]

    def check_params(self, params_all, params_free):
        if self.nz_analytic:
            if 'width' in params_free:
                raise KeyError("Using analytic N(z). Width is irrelevant")
            if ('z_tail' not in params_all) or \
               ('z_ini' not in params_all) or \
               ('gamma' not in params_all):
                raise KeyError("Using analytic N(z). Need z_tail, z_ini and gamma")
        else:
            if ('z_tail' in params_free) or \
               ('z_ini' in params_free) or \
               ('gamma' in params_free):
                raise KeyError("Using template N(z). z_tail, z_ini and gamma irrelevant")
            if 'width' not in params_all:
                raise KeyError("Using template N(z). Need width")
        if ('bias' not in params_all) or \
           ('sigma8' not in params_all):
            raise KeyError("Need bias and sigma8")

    def get_nz(self, **kwargs):
        if self.nz_analytic:
            zs = self.z
            z_o_ini_p = (zs / kwargs['z_ini'])**2
            z_o_tail_p = (zs / kwargs['z_tail'])**kwargs['gamma']
            nz = z_o_ini_p/((1+z_o_ini_p)*(1+z_o_tail_p))
        else:
            zs = kwargs['width']*self.z
            nz = self.nz
        return zs, nz
        
    def get_cl(self, ell, cl_types, **kwargs):
        self._get_cosmo(kwargs['sigma8'])
        zs, nz = self.get_nz(**kwargs)
        if self.bias_type == 'constant':
            bzs = kwargs['bias']*np.ones_like(zs)
        elif self.bias_type == 'inv_growth':
            bzs = kwargs['bias']/ccl.growth_factor(self.cosmo, 1./(1+zs))
        elif self.bias_type == 'plateau':
            zr2 = (zs/1.5)**2
            bzs = kwargs['bias']*(1+2*zr2)/(1+zr2)
        else:
            raise ValueError("Unknown bias type %s" % bias_type)
        trs = {}
        trs['g'] = ccl.NumberCountsTracer(self.cosmo, False,
                                         (zs, nz),
                                         (zs, bzs))
        if 'gk' in cl_types:
            trs['k'] = ccl.CMBLensingTracer(self.cosmo, z_source=1100.)

        t = []
        for typ in cl_types:
            t1 = trs[typ[0]]
            t2 = trs[typ[1]]
            t.append(ccl.angular_cl(self.cosmo, t1, t2, ell))
        return np.array(t)

    def _get_cosmo(self, sigma8):
        if self.sigma8 != sigma8:
            self.sigma8 = sigma8
            self.cosmo = ccl.Cosmology(Omega_c=0.26066676,
                                       Omega_b=0.048974682,
                                       h=0.6766,
                                       sigma8=sigma8,
                                       n_s=0.9665)


class Like(object):
    def __init__(self, config):
        self.prefix_out = config['output_prefix']

        self.cl_list = config['cls_include']

        # Read Cls
        f = np.load(config['cl_file'])
        #  - ell
        self.l = f['l_eff']
        self.lmax = config['l_max']
        msk = self.l < self.lmax
        self.l = self.l[msk]

        #  - Cls
        self.dv = []
        self.nv = []
        for k in self.cl_list:
            self.dv.append((f['cl_' + k] - f['nl_' + k])[msk])
            self.nv.append(f['nl_' + k][msk])
        self.dv = np.array(self.dv).flatten()
        self.nv = np.array(self.nv).flatten()

        #  - Covariance
        self.nd_tot = len(self.dv)
        self.nd_single = np.sum(msk)
        self.n_cls = len(self.cl_list)
        self.cv = np.zeros([self.nd_tot, self.nd_tot])
        for i1, k1 in enumerate(self.cl_list):
            for i2, k2 in enumerate(self.cl_list):
                if 'cov_'+k1+k2 in f:
                    cov = f['cov_'+k1+k2]
                elif 'cov_'+k2+k1 in f:
                    cov = f['cov_'+k2+k1].T
                else:
                    raise ValueError("Can't find covariance "+k1+k2)
                cov = cov[msk][:, msk]
                self.cv[i1*self.nd_single:(i1+1)*self.nd_single][:,i2*self.nd_single:(i2+1)*self.nd_single] = cov
        self.ic = np.linalg.inv(self.cv)
        self.cvhalf = np.linalg.cholesky(self.cv)
        
        self.p_free_names = []
        self.p_free_labels = []
        self.p_free_prior = []
        self.p_fixed = []
        self.p0 = []
        pars = config['params']
        for p in pars:
            n = p.get('name')
            if not p['vary']:
                self.p_fixed.append((n, p.get('value')))
            else:
                self.p_free_names.append(n)
                self.p_free_labels.append(p.get('label'))
                self.p_free_prior.append(p.get('prior'))
                self.p0.append(p.get('value'))
                                                                
        
        self.model = Model(config['nz']['nz_source'], self.l,
                           z_name=config['nz'].get('z_name', 'z_g'),
                           nz_name=config['nz'].get('nz_name', 'nz_g_w'),
                           bias_type=config['bias'].get('bias_type', 'constant'))
        self.model.check_params(list(dict(self.p_fixed).keys()) + self.p_free_names,
                                self.p_free_names)

        # Emcee params
        self.nwalkers = config['sampler']['nwalkers']
        self.ndim = len(self.p0)
        self.nsteps = config['sampler']['nsteps']
        self.rerun = config['sampler']['rerun']

    def update_p0(self, p):
        self.p0 = p

    def get_best_fit(self, p0=None,
                     xtol=0.0001, ftol=0.0001, maxiter=None,
                     options=None, update_p0=False):
        from scipy.optimize import minimize

        # Initial parameters
        if p0 is None:
            p0 = self.p0

        # Minimizer options
        opt = {'xtol': xtol, 'ftol': ftol, 'maxiter': maxiter}
        if options is not None:
            opt.update(options)

        # Run minimizer
        with warnings.catch_warnings():  # Suppress warnings due to np.inf
            warnings.simplefilter("ignore")
            res = minimize(self.chi2, p0, method="Powell", options=opt)

        # Update if needed
        if update_p0:
            self.update_p0(np.atleast_1d(res.x))
        return np.atleast_1d(res.x)

    def build_kwargs(self, par):
        params = dict(self.p_fixed)
        params.update(dict(zip(self.p_free_names, par)))
        return params

    def get_data(self, split=False):
        if split:
            return self.dv.reshape([self.n_cls,
                                    self.nd_single])
        else:
            return self.dv

    def get_noise(self, split=False):
        if split:
            return self.nv.reshape([self.n_cls,
                                    self.nd_single])
        else:
            return self.nv

    def get_covar(self, split=False):
        if split:
            return self.cv.reshape([self.n_cls, self.nd_single,
                                    self.n_cls, self.nd_single])
        else:
            return self.cv

    def lnprior(self, par):
        lnp = 0
        for p, pr in zip(par, self.p_free_prior):
            if pr is None:  # No prior
                continue
            elif pr['type'] == 'Gaussian':
                lnp += -0.5 * ((p - pr['values'][0]) / pr['values'][1])**2
            else:
                if not(pr['values'][0] <= p <= pr['values'][1]):
                    return -np.inf
        return lnp

    def lnlike(self, par):
        tv = self.get_theory(par)
        if tv is None:  # Theory calculation failed
            return -np.inf
        dx = self.dv-tv
        return -0.5 * np.einsum('i,ij,j', dx, self.ic, dx)

    def lnprob(self, par):
        pr = self.lnprior(par)
        if pr != -np.inf:
            pr += self.lnlike(par)
        return pr

    def chi2(self, pars):
        return -2*self.lnprob(pars)

    def generate_data(self, par):
        params = self.build_kwargs(par)
        tv = self.get_theory(params)
        return tv+np.dot(self.cvhalf, np.random.randn(len(tv)))

    def get_theory(self, par, split=False, hires=False):
        try:
            params = self.build_kwargs(par)
            if hires:
                l_use = np.arange(10, self.lmax)
            else:
                l_use = self.l
            t = self.model.get_cl(l_use, self.cl_list, **params)
            if split:
                return t
            else:
                return t.flatten()
        except:
            print("Encountered theory error at ", par)
            return None

    def get_Fisher(self, p0=None, minimize=False):
        import numdifftools as nd
        if minimize:
            p0 = self.get_best_fit(p0)
        if p0 is None:
            p0 = self.p0
        def lnprobd(p):
            l = self.lnprob(p)
            if l == -np.inf:
                l = -1E100
            return l
        fisher = - nd.Hessian(lnprobd)(p0)
        return fisher

    def get_Fisher_covariance(self, p0=None, minimize=False):
        fisher = self.get_Fisher(p0, minimize)
        return np.linalg.inv(fisher)

    def sample(self, verbosity=0, use_mpi=False):
        import emcee
        if use_mpi:
            from schwimmbad import MPIPool
            pool = MPIPool()
            print("Using MPI")
            pool_use = pool
        else:
            pool = DumPool()
            print("Not using MPI")
            pool_use = None

        if not pool.is_master():
            pool.wait()
            sys.exit(0)

        fname_chain = self.prefix_out+"chain"
        found_file = os.path.isfile(fname_chain+'.txt')

        if (not found_file) or self.rerun:
            pos_ini = (np.array(self.p0)[None, :] +
                       0.001 * np.random.randn(self.nwalkers, self.ndim))
            nsteps_use = self.nsteps
        else:
            print("Restarting from previous run")
            old_chain = np.loadtxt(fname_chain+'.txt')
            if np.ndim(old_chain) == 1:
                old_chain = np.atleast_2d(old_chain).T
            pos_ini = old_chain[-self.nwalkers:, :]
            nsteps_use = max(self.nsteps-len(old_chain) // self.nwalkers, 0)
            print(self.nsteps - len(old_chain) // self.nwalkers)

        chain_file = SampleFileUtil(self.prefix_out+"chain", rerun=self.rerun)
        sampler = emcee.EnsembleSampler(self.nwalkers,
                                        self.ndim,
                                        self.lnprob,
                                        pool=pool_use)
        counter = 1
        for pos, prob, _ in sampler.sample(pos_ini, iterations=nsteps_use):
            if pool.is_master():
                chain_file.persistSamplingValues(pos, prob)

                if counter % 10 == 0:
                    print(f"Finished sample {counter}")
            counter += 1

        pool.close()

        return sampler

    def get_chain(self):
        """
        Read chain from previous run. Chain can be retireved in the `chain`
        attribute. The log-posterior for each sample can be retrieved through
        the `probs` attribute.
        """
        self.chain = np.loadtxt(self.prefix_out + "chain.txt")
        if np.ndim(self.chain) == 1:
            self.chain = np.atleast_2d(self.chain).T
        self.probs = np.loadtxt(self.prefix_out + "chainprob.txt")

    def plot_chain(self, chain, save_figure=False, prefix=None,
                   extension='pdf'):
        """
        Produces a triangle plot from a chain, which can be
        saved to file automatically.

        Args:
            chain (array): 2D array with shape [n_samples,n_params],
                where `n_samples` is the number of samples in the
                chain and `n_params` is the number of free parameters
                for this likelihood run.
            save_figures (bool): if true, figures will be saved to
                file. File names will take the form:
                <`prefix`>triangle.<`extension`>
            prefix (str): output prefix.
            extension (str): plot extension (pdf, pdf etc.).

        Returns:
            figure object
        """
        from getdist import MCSamples
        from getdist import plots as gplots

        nsamples = len(chain)
        # Generate samples
        ranges={}
        for n,pr in zip(self.p_free_names, self.p_free_prior):
            if pr['type']=='TopHat':
                ranges[n]=pr['values']
        samples = MCSamples(samples=chain[nsamples//4:],
                            names=self.p_free_names,
                            labels=self.p_free_labels,
                            ranges=ranges)
        samples.smooth_scale_2D=0.2

        # Triangle plot
        g = gplots.getSubplotPlotter()
        g.triangle_plot([samples], filled=True)

        if save_figure:
            if prefix is None:
                prefix = self.prefix_out
            fname = prefix+'triangle.'+extension
            g.export(fname)

        return g
