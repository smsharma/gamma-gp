import sys

sys.path.append("../")

from pathlib import Path

import numpy as np
import healpy as hp
import torch
import gpytorch
import pyro.distributions as dist
from scipy.stats import poisson
from scipy.optimize import minimize

import utils.create_mask as cm
from utils.psf_correction import PSFCorrection
from utils.utils import make_dirs
from models.psf import KingPSF
from models.scd import dnds
from models.gp_regression import HealpixGPRegressionModel
from simulations.simulate_ps import SimulateMap


class ModelConstructor:
    def __init__(self, data_dir=None, r_outer=20.0, nside=128, theta_poiss=[0.5, 0.1, 0.5, 0.0, 8.0, 4.0], theta_ps=[1.5, 20.0, 1.8, -20.0, 20.0, 0.1], num_inducing=200, guide_name="MVN", n_exp=1, dif_sim="mO", dif_fit="p6", gp_mean="zero", run_name="test", save_dir=None, kernel="matern52", mask_ps=True, mask_plane=True, outputscale_prior=None, lengthscale_prior=None, learn_inducing_locations=True, inducing_strategy="uniform", poiss_only=True, plot_inducing=True):
        """ High-level interface to construct Pyro model with GP component.

            :param data_dir: Directory where data is stored. Defaults to '../data/'.
            :param r_outer: Outer radius of mask. Defaults to 20.
            :param nside: Healpix nside parameter. Defaults to 128.
            :param theta_poiss: Norms of Poissonian templates ["bub", "iso", "psc", "gce", "dif_pibrem", "dif_ics"]. Defaults to [0.5, 0.1, 0.5, 0.0, 8.0, 4.0].
            :param theta_ps: Non-Poissonian GCE template parameters for 2-break SCD, NPTFit convention. Defaults to [1.5, 20.0, 1.8, -20.0, 20.0, 0.1].
            :param num_inducing: Number of variational GP inducing points. Defaults to 200.
            :param guide_name: Guide type for template parameters. Defaults to "MVN".
                "MVN": Multivariate normal guide
                "IAF": Inverse autoregressive flow guide
                "ConditionalIAF": Inverse autoregressive flow guide with NN conditioned on GP summary parameters
            :param n_exp: Number of exposure regions for NPTF likelihood. Defaults to 1.
            :param dif_sim: Diffuse model used for simulation. "Model O" hard-coded at the moment. Defaults to "mO".
            :param dif_fit: Diffuse model used for fit. "p6" hard-coded at the moment. Defaults to "p6".
            :param gp_mean: GP mean specification, ["zero", "constant"]. Defaults to "zero".
            :param run_name: Tag of run. Defaults to "test".
            :param save_dir: Directory in which to save model state. By default `save` in current dir. Defaults to None.
            :param kernel: GP kernel. One of ["matern52", "matern32", "matern12", "rbf]. Defaults to "matern52".
            :param mask_ps: Whether to mask resolved PSs in ROI. Defaults to True.
            :param mask_plane: Whether to mask the plane in ROI. Defaults to True.
            :param outputscale_prior: (mean, variance) of outputscale prior of GP kernel. Defaults to None.
            :param lengthscale_prior: (mean, variance) of lengthscale prior of GP kernel. Defaults to None.
            :param learn_inducing_locations: Whether to learn positions of inducing points. Defaults to True.
            :param inducing_strategy: Inducing points specification, either "uniform" or "all", latter setting all ROI pixels as inducing points. Defaults to "uniform".
            :param poiss_only: Whether to use only Poissonian likelihood (no PS templates). Defaults to True.
            :param plot_inducing: Whether to plot a healpix map of inducing point locations. Defaults to True.
        """
        self.nside = nside
        self.num_inducing = num_inducing
        self.guide_name = guide_name
        self.n_exp = n_exp
        self.r_outer = r_outer
        self.theta_ps = theta_ps
        self.theta_poiss = theta_poiss
        self.gp_mean = gp_mean
        self.kernel = kernel

        self.poiss_only = poiss_only

        self.inducing_strategy = inducing_strategy
        self.learn_inducing_locations = learn_inducing_locations

        self.outputscale_prior = outputscale_prior
        self.lengthscale_prior = lengthscale_prior

        self.mask_ps = mask_ps
        self.mask_plane = mask_plane

        self.dif_fit = dif_fit
        self.dif_sim = dif_sim

        self.run_name = run_name
        self.plot_inducing = plot_inducing

        self.n_pix = hp.nside2npix(self.nside)

        if self.lengthscale_prior is not None:
            mu, sigma = self.lengthscale_prior
            self.lengthscale_prior = gpytorch.priors.NormalPrior(mu, sigma)

        if self.outputscale_prior is not None:
            mu, sigma = self.outputscale_prior
            self.outputscale_prior = gpytorch.priors.NormalPrior(mu, sigma)

        if not data_dir:
            data_dir = str(Path(__file__).parent / "../data/")

        self.save_dir = save_dir
        if not self.save_dir:
            self.save_dir = str(Path(__file__).parent / "../inference/save/")
        self.save_dir += "/" + run_name + "/"
        make_dirs([self.save_dir])

        self.load_data(data_dir)
        self.construct_mask()
        self.load_psf()
        self.simulate()
        self.do_scipy_fit(self.data)
        self.construct_model()

    def do_scipy_fit(self, data):
        """ SciPy fit to data to get norms of templates
        """

        def log_like(theta, data):
            """ Bin-wise Poisson likelihood
            """

            # The parameters are the overall norms of each template
            A_bub, A_iso, A_psc, A_gce, A_dif = theta

            # Model is the sum of templates with free normalizations
            mu = (A_bub * self.temp_bub + A_iso * self.temp_iso + A_psc * self.temp_psc + A_gce * self.temp_gce + A_dif * self.temp_dif_fit)[~self.mask]

            # Bin-wise Poisson sum
            return np.sum(poisson.logpmf(data, mu))

        self.opt = minimize(lambda theta: -log_like(theta, data[~self.mask]), x0=[1.0] * 5, bounds=[[0, 30]] * 5, method="L-BFGS-B")

    def load_data(self, data_dir):
        """ Load all the data and templates
        """

        # Load data and exposure
        self.fermi_counts = hp.ud_grade(np.load(data_dir + "/fermi_data/fermidata_counts.npy"), nside_out=self.nside, power=-2)
        self.fermi_exp = hp.ud_grade(np.load(data_dir + "/fermi_data/fermidata_exposure.npy"), nside_out=self.nside, power=-2)

        # Load templates
        self.temp_bub = hp.ud_grade(np.load(data_dir + "/fermi_data/template_bub.npy"), nside_out=self.nside, power=-2)
        self.temp_dsk = hp.ud_grade(np.load(data_dir + "/fermi_data/template_dsk.npy"), nside_out=self.nside, power=-2)
        self.temp_psc = hp.ud_grade(np.load(data_dir + "/fermi_data/template_psc.npy"), nside_out=self.nside, power=-2)
        self.temp_iso = hp.ud_grade(np.load(data_dir + "/fermi_data/template_iso.npy"), nside_out=self.nside, power=-2)
        self.temp_gce = hp.ud_grade(np.load(data_dir + "/fermi_data/template_gce.npy"), nside_out=self.nside, power=-2)
        self.temp_p6 = hp.ud_grade(np.load(data_dir + "/fermi_data/template_dif.npy"), nside_out=self.nside, power=-2)

        # Model O
        self.temp_mO_ics = hp.ud_grade(np.load(data_dir + "/fermi_data/ModelO_r25_q1_ics.npy"), nside_out=self.nside, power=-2)
        self.temp_mO_pibrem = hp.ud_grade(np.load(data_dir + "/fermi_data/ModelO_r25_q1_pibrem.npy"), nside_out=self.nside, power=-2)

        self.temp_m0_tot = self.temp_mO_ics + self.temp_mO_pibrem

        # Model A
        self.temp_mA_ics = hp.ud_grade(np.load(data_dir + "/modelA/modelA_ics.npy"), nside_out=self.nside, power=-2)
        self.temp_mA_pibrem = hp.ud_grade(np.load(data_dir + "/modelA/modelA_brempi0.npy"), nside_out=self.nside, power=-2)

        if self.dif_sim == "p6":
            self.temp_dif_sim = self.temp_p6
        elif self.dif_sim == "m0":
            self.temp_dif_sim = self.temp_m0_tot

        if self.dif_fit == "p6":
            self.temp_dif_fit = self.temp_p6
        elif self.dif_fit == "m0":
            self.temp_dif_fit = self.temp_m0_tot

        # PS templates
        self.temp_gce_ps = self.temp_gce / self.fermi_exp
        self.temp_gce_ps /= np.mean(self.temp_gce_ps)

        self.temp_dsk_ps = self.temp_dsk / self.fermi_exp
        self.temp_dsk_ps /= np.mean(self.temp_dsk_ps)

        # Load PS mask
        self.ps_mask = hp.ud_grade(np.load(data_dir + "/mask_3fgl_0p8deg.npy") > 0, nside_out=self.nside)

    def construct_mask(self):
        """ Construct mask with or without masking PSs
        """
        if self.mask_ps:
            self.mask = cm.make_mask_total(nside=self.nside, band_mask=self.mask_plane, band_mask_range=2, mask_ring=True, inner=0, outer=self.r_outer, custom_mask=self.ps_mask)
        else:
            self.mask = cm.make_mask_total(nside=self.nside, band_mask=self.mask_plane, band_mask_range=2, mask_ring=True, inner=0, outer=self.r_outer)

    def load_psf(self):
        """ Load Fermi PSF, use hard-coded King PSF at 2 GeV
        """

        self.kp = KingPSF()

        pc_inst = PSFCorrection(delay_compute=True)
        pc_inst.psf_r_func = lambda r: self.kp.psf_fermi_r(r)
        pc_inst.sample_psf_max = 10.0 * self.kp.spe * (self.kp.score + self.kp.stail) / 2.0
        pc_inst.psf_samples = 10000
        pc_inst.psf_tag = "Fermi_PSF_2GeV"
        pc_inst.make_or_load_psf_corr()

        self.f_ary = pc_inst.f_ary
        self.df_rho_div_f_ary = pc_inst.df_rho_div_f_ary

    def simulate(self):
        """ Simulate map with point sources
        """

        s_ary = torch.logspace(-2, 2, 1000)  # Range of counts considered
        theta_ps_sim = torch.tensor(self.theta_ps).clone()
        theta_ps_sim[0] *= self.n_pix  # Normalization of source-count distribution amplitude

        dnds_ary = dnds(s_ary, theta_ps_sim)  # Get source-count distribution corresponding to PS parameters

        psf_r_func = lambda r: self.kp.psf_fermi_r(r)  # Fermi King PSF hard-coded

        # Use either "p6" or "mO" diffuse model as specified, other templates always the same
        temp_sim_list = []
        if self.dif_sim == "p6":
            temp_sim_list = [self.temp_bub, self.temp_iso, self.temp_psc, self.temp_gce, self.temp_p6]
        elif self.dif_sim == "mO":
            temp_sim_list = [self.temp_bub, self.temp_iso, self.temp_psc, self.temp_gce, self.temp_mO_pibrem, self.temp_mO_ics]

        self.sim = SimulateMap(temp_sim_list, self.theta_poiss, [s_ary], [dnds_ary.detach().numpy()], [self.temp_gce_ps], psf_r_func, nside=self.nside)

        self.data = self.sim.create_map()

    def construct_model(self):

        # Define angular coordinates and counts data
        X = torch.tensor(np.radians(hp.pix2ang(self.nside, np.arange(hp.nside2npix(self.nside)), lonlat=True))).float().T
        Y = torch.tensor(self.data).float()

        # Mask to get "training" data
        self.train_x = X[~self.mask]
        self.train_y = Y[~self.mask]

        # If Poisson regression, no PS
        if self.poiss_only:
            ps_temps = []
        else:
            ps_temps = [self.temp_gce_ps, self.temp_dsk_ps]

        # Instantiate Pyro model to be trained
        self.model = HealpixGPRegressionModel(
            exposure_map=self.fermi_exp,
            nexp=self.n_exp,
            temp_dif=self.opt.x[-1] * self.temp_dif_fit,
            poiss_temps=[self.temp_bub, self.temp_iso, self.temp_psc, self.temp_gce],
            poiss_labels=["bub", "iso", "psc", "gce"],
            poiss_priors=[dist.Normal(torch.tensor(0.5), torch.tensor(0.1)), dist.Normal(torch.tensor(0.1), torch.tensor(0.02)), dist.Normal(torch.tensor(0.5), torch.tensor(0.1)), dist.Uniform(torch.tensor(0.0), torch.tensor(1.5))],
            poiss_log_priors=[0.0, 0.0, 0.0, 0.0],
            ps_temps=ps_temps,
            ps_labels=["gce", "dsk"],
            ps_priors=2 * [[dist.Uniform(torch.tensor(0.0), torch.tensor(0.5)), dist.Uniform(torch.tensor(11.0), torch.tensor(20.0)), dist.Uniform(torch.tensor(1.1), torch.tensor(1.99)), dist.Uniform(torch.tensor(-10.0), torch.tensor(1.99)), dist.Uniform(torch.tensor(1.0), torch.tensor(40.0)), dist.Uniform(torch.tensor(0.01), torch.tensor(0.1))]],
            ps_log_priors=torch.tensor(2 * [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            relative_breaks=True,
            mask=self.mask,
            name_prefix=self.run_name,
            num_inducing=self.num_inducing,
            nside=self.nside,
            guide_name=self.guide_name,
            no_gp=False,
            f_ary=self.f_ary,
            df_rho_div_f_ary=self.df_rho_div_f_ary,
            gp_mean=self.gp_mean,
            kernel=self.kernel,
            lengthscale_prior=self.lengthscale_prior,
            outputscale_prior=self.outputscale_prior,
            inducing_strategy=self.inducing_strategy,
            learn_inducing_locations=self.learn_inducing_locations,
            plot_inducing=self.plot_inducing,
        )

