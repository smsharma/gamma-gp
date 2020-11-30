import sys

sys.path.append("../")

import numpy as np
import healpy as hp

import torch
import gpytorch
from gpytorch.models import ApproximateGP
import pyro

from models.kernels import CustomRBFKernel, CustomMatern12Kernel, CustomMatern32Kernel, CustomMatern52Kernel
from models.template_param import TemplateParamModel
from models.likelihoods import log_like_np
from utils.utils import fibonacci_sphere, intersect1d


class HealpixGPRegressionModel(ApproximateGP, TemplateParamModel):
    """ Sparse variational Gaussian process regression with non-Poissonian template likelihoods
    """

    def __init__(self, exposure_map, temp_dif, poiss_temps=[], poiss_labels=[], poiss_priors=[], ps_temps=[], ps_labels=[], ps_priors=[], relative_breaks=False, mask=None, f_ary=[1.0], df_rho_div_f_ary=[1.0], poiss_log_priors=None, ps_log_priors=None, num_inducing=300, inducing_strategy="uniform", learn_inducing_locations=True, nside=128, name_prefix="test", guide_name="MVN", guide_kwargs={}, no_gp=False, nexp=1, plot_inducing=True, gp_mean="zero", kernel="matern52", lengthscale_prior=None, outputscale_prior=None):

        # Instantiate templates class

        TemplateParamModel.__init__(self, exposure_map, nexp, poiss_temps, poiss_labels, poiss_priors, ps_temps, ps_labels, ps_priors, mask=mask, f_ary=f_ary, df_rho_div_f_ary=df_rho_div_f_ary, poiss_log_priors=poiss_log_priors, ps_log_priors=ps_log_priors, relative_breaks=relative_breaks, name_prefix=name_prefix, guide_name=guide_name)

        self.name_prefix = name_prefix
        self.no_gp = no_gp
        self.nexp = nexp

        if mask is not None:
            self.temp_dif = torch.tensor(temp_dif[~mask])
        else:
            self.temp_dif = torch.tensor(temp_dif)

        # Define all the variational stuff

        # Uniformly-distributed initial inducing points
        if inducing_strategy == "uniform":
            masked_fraction = np.sum(~mask) / hp.nside2npix(nside)
            num_inducing_allsky = np.ceil(num_inducing / masked_fraction).astype(np.int32)
            mask_pixels = np.where(~mask == True)
            inducing_pixels = hp.vec2pix(nside, *np.transpose(np.array(fibonacci_sphere(num_inducing_allsky))))
            inducing_points = torch.tensor(np.radians(hp.pix2ang(nside, np.intersect1d(inducing_pixels.astype(np.int32), mask_pixels), lonlat=True))).float().T
            temp = np.zeros(hp.nside2npix(nside))
            temp[inducing_pixels] = 1.0

            # View inducing points
            if plot_inducing:
                hp.mollview(~mask * temp, title="Inducing point locations", cmap="Greys_r")

        # Initial inducing points == training points
        elif inducing_strategy == "all":
            inducing_pixels = np.arange(hp.nside2npix(nside))[~mask]
            mask_pixels = np.where(~mask == True)
            inducing_points = torch.tensor(np.radians(hp.pix2ang(nside, np.intersect1d(inducing_pixels.astype(np.int32), mask_pixels), lonlat=True))).float().T
        else:
            raise RuntimeError("Invalid inducing strategy.")

        variational_strategy = gpytorch.variational.VariationalStrategy(self, inducing_points, gpytorch.variational.CholeskyVariationalDistribution(num_inducing_points=len(inducing_points)), learn_inducing_locations=learn_inducing_locations)

        # Standard initializtation
        super().__init__(variational_strategy)

        # GP mean and covariance

        if gp_mean == "zero":
            self.mean_module = gpytorch.means.ZeroMean()
        elif gp_mean == "constant":
            self.mean_module = gpytorch.means.ConstantMean()
        else:
            raise NotImplementedError

        if kernel == "matern12":
            self.covar_module = gpytorch.kernels.ScaleKernel(CustomMatern12Kernel(lengthscale_prior=lengthscale_prior), outputscale_prior=outputscale_prior)
        elif kernel == "matern32":
            self.covar_module = gpytorch.kernels.ScaleKernel(CustomMatern32Kernel(lengthscale_prior=lengthscale_prior), outputscale_prior=outputscale_prior)
        elif kernel == "matern52":
            self.covar_module = gpytorch.kernels.ScaleKernel(CustomMatern52Kernel(lengthscale_prior=lengthscale_prior), outputscale_prior=outputscale_prior)
        elif kernel == "rbf":
            self.covar_module = gpytorch.kernels.ScaleKernel(CustomRBFKernel(lengthscale_prior=lengthscale_prior), outputscale_prior=outputscale_prior)
        else:
            raise NotImplementedError

        # self.num_data = len(self.temp_dif)

        self.mask = torch.tensor(mask)

        # Initial IAF guide
        if (self.guide_name in ["ConditionalIAF", "IAF"]) and (self.n_params > 0):
            self.init_iaf_guide(**guide_kwargs)
        elif (self.guide_name == "MVN") and (self.n_params > 0):
            self.init_mvn_guide()

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean, covar)

    def guide(self, x, y, indices):
        """ Guide for GP and template parameters
        """

        if not self.no_gp:
            # Get q(f) - variational (guide) distribution of latent function
            function_dist = self.pyro_guide(x)

            # Use a plate here to mark conditional independencies
            with pyro.plate(self.name_prefix + ".data_plate", dim=-1):
                # Sample from latent function distribution
                gp_guide_sample = pyro.sample(self.name_prefix + ".f(x)", function_dist)
        else:
            gp_guide_sample = None

        # Guide for template parameters

        if self.n_params > 0:
            if self.guide_name in ["ConditionalIAF", "IAF"]:
                self.templates_guide_iaf(gp_sample=gp_guide_sample, indices=indices)
            elif self.guide_name == "MVN":
                self.templates_guide_mvn()

    def model(self, x, y, indices):
        """ GP model with non-Poissonian likelihood
        """

        pyro.module(self.name_prefix + ".gp", self)

        # Sample Poissonian norms and create Poissonian expectation

        mu_poiss = torch.zeros(torch.sum(~self.mask), dtype=torch.float64)

        for i_temp in torch.arange(self.n_poiss):
            norms_poiss = pyro.sample(self.poiss_labels[i_temp], self.poiss_priors[i_temp])
            if self.poiss_log_priors[i_temp]:
                norms_poiss = 10 ** norms_poiss.clone()
            mu_poiss += norms_poiss * self.poiss_temps[i_temp]

        # Hacky---set negative pixels to zero
        mu_poiss[mu_poiss < 0.0] = 0.0

        # Samples non-Poissonian parameters

        thetas = []

        for i_ps in torch.arange(self.n_ps):
            theta_temp = [pyro.sample(self.ps_param_labels[i_np_param] + "_" + self.ps_labels[i_ps], self.ps_priors[i_ps][i_np_param]) for i_np_param in torch.arange(self.n_ps_params)]

            for i_p in torch.arange(self.n_ps_params):
                if self.ps_log_priors[i_ps][i_p]:
                    theta_temp[i_p] = 10 ** theta_temp[i_p]

            # Hard-coded relative breaks for doubly-broken power law
            if self.relative_breaks:
                theta_temp[-1] = theta_temp[-1] * theta_temp[-2]

            thetas.append(theta_temp)

        if not self.no_gp:

            # Use the exponential link function to convert GP samples into scale samples

            # Get p(f) - prior distribution of latent function
            function_dist = self.pyro_model(x)

            # Sample from latent function distribution
            function_samples = pyro.sample(self.name_prefix + ".f(x)", function_dist)

            # Modulate diffuse template by GP and add to Poissonian expectation
            temp_dif_gp = self.temp_dif[indices] * function_samples.exp()
            mu = mu_poiss[indices] + temp_dif_gp
        else:
            mu = mu_poiss[indices]

        # Use a plate here to mark conditional independencies

        with pyro.plate(self.name_prefix + ".data_plate", dim=-1):

            # If no PS model, use simple Poissonian likelihood
            if self.n_ps == 0:
                log_likelihood = (mu.log() * y) - mu - (y + 1).lgamma()

            # Use non-Poissonian likelihood
            else:

                # Get indices of exposure regions within ROI
                expreg_sub_indices = [intersect1d(indices, self.expreg_indices[i]) for i in torch.arange(self.nexp)]

                mu_embed = torch.zeros(len(self.temp_dif), dtype=torch.float64)
                y_embed = torch.zeros(len(self.temp_dif))

                mu_embed[indices] = mu.clone()
                y_embed[indices] = y.clone()

                log_likelihood_expregs = []

                # Compute separate log-likelihood for each exposure region, scaling the non-Poissonian parameters appropriately
                for i_exp in torch.arange(self.nexp):

                    exposure_multiplier = self.exposure_means_list[i_exp] / self.exposure_mean

                    # Scale non-Poissonian parameters (norm divided by exposure ratio, breaks multiplied)
                    thetas_expreg = thetas

                    for i_p in torch.arange(self.n_ps):
                        thetas_expreg[i_p][0] = thetas_expreg[i_p][0] / exposure_multiplier
                        thetas_expreg[i_p][-1] = thetas_expreg[i_p][-1] * exposure_multiplier
                        thetas_expreg[i_p][-2] = thetas_expreg[i_p][-2] * exposure_multiplier

                    log_likelihood_expreg = log_like_np(mu_embed[expreg_sub_indices[i_exp]], thetas_expreg, self.ps_temps[:, expreg_sub_indices[i_exp]], y_embed[expreg_sub_indices[i_exp]], self.f_ary, self.df_rho_div_f_ary)
                    log_likelihood_expregs.append(log_likelihood_expreg)

                # Combine likelihoods from exposure regions to get final (pixel-wise) log-likelihood
                log_likelihood = torch.cat(log_likelihood_expregs)

            pyro.factor(self.name_prefix + ".log_likelihood", log_likelihood)
