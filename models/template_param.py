import functools
import torch
from torch import nn
from torch.distributions import biject_to, constraints
import pyro
from pyro.nn import PyroParam, PyroModule, PyroSample
from pyro.distributions.util import eye_like, sum_rightmost
import pyro.distributions as dist
from pyro.distributions.transforms import conditional_affine_autoregressive, affine_autoregressive, iterated
import numpy as np
import healpy as hp


class TemplateParamModel:
    def __init__(self, exposure_map, nexp, poiss_temps, poiss_labels, poiss_priors, ps_temps=[], ps_labels=[], ps_priors=[], mask=None, f_ary=[1.0], df_rho_div_f_ary=[1.0], poiss_log_priors=None, ps_log_priors=None, relative_breaks=False, name_prefix="", guide_name="MVN"):

        # Parameter bookkeeping

        self.poiss_temps = torch.tensor(poiss_temps)
        self.poiss_labels = poiss_labels
        self.poiss_priors = poiss_priors

        self.ps_temps = torch.tensor(ps_temps)
        self.ps_labels = ps_labels
        self.ps_priors = ps_priors
        self.relative_breaks = relative_breaks

        self.f_ary = torch.tensor(f_ary, dtype=torch.float64)
        self.df_rho_div_f_ary = torch.tensor(df_rho_div_f_ary, dtype=torch.float64)

        self.name_prefix = name_prefix

        self.guide_name = guide_name

        # Some bookkeeping for numbers of parameters

        self.n_poiss = len(self.poiss_temps)
        self.n_ps = len(self.ps_temps)

        self.ps_param_labels = ["A_ps", "n_1", "n_2", "n_3", "Sb_1", "Sb_2"]
        self.n_ps_params = 6

        self.n_params = self.n_poiss + self.n_ps_params * self.n_ps

        self.exposure_map = exposure_map

        # No log-linear priors by defaults

        if poiss_log_priors is None:
            self.poiss_log_priors = torch.zeros(self.n_poiss)
        else:
            self.poiss_log_priors = poiss_log_priors

        if ps_log_priors is None:
            self.ps_log_priors = torch.zeros((self.n_ps, 6))
        else:
            self.ps_log_priors = ps_log_priors

        # Apply mask to templates if applicable
        if mask is not None:
            self.mask = torch.tensor(mask, dtype=torch.bool)
            self.mask_templates()

        self.get_exp_regions(nexp)

        self.n_pix = torch.sum(~self.mask)

        if self.guide_name not in ["ConditionalIAF", "IAF", "MVN"]:
            raise NotImplementedError("Guide not implemented!")

    def mask_templates(self):

        poiss_temps = torch.zeros((self.n_poiss, torch.sum(~self.mask)), dtype=torch.float64)
        for i_temp in range(self.n_poiss):
            poiss_temps[i_temp, :] = self.poiss_temps[i_temp, ~self.mask]
        ps_temps = torch.zeros((self.n_ps, torch.sum(~self.mask)), dtype=torch.float64)
        for i_temp in range(self.n_ps):
            ps_temps[i_temp, :] = self.ps_temps[i_temp, ~self.mask]

        self.poiss_temps = poiss_temps
        self.ps_temps = ps_temps

    def templates_guide_mvn(self):
        """ Multivariate normal guide for template parameters
        """

        loc = _deep_getattr(self, "mvn.loc")
        scale_tril = _deep_getattr(self, "mvn.scale_tril")

        dt = dist.MultivariateNormal(loc, scale_tril=scale_tril)
        states = pyro.sample("states_" + self.name_prefix, dt, infer={"is_auxiliary": True})

        result = {}

        for i_poiss in torch.arange(self.n_poiss):
            transform = biject_to(self.poiss_priors[i_poiss].support)
            value = transform(states[i_poiss])
            log_density = transform.inv.log_abs_det_jacobian(value, states[i_poiss])
            log_density = sum_rightmost(log_density, log_density.dim() - value.dim() + self.poiss_priors[i_poiss].event_dim)

            result[self.poiss_labels[i_poiss]] = pyro.sample(self.poiss_labels[i_poiss], dist.Delta(value, log_density=log_density, event_dim=self.poiss_priors[i_poiss].event_dim))

        i_param = self.n_poiss

        for i_ps in torch.arange(self.n_ps):
            for i_ps_param in torch.arange(self.n_ps_params):

                transform = biject_to(self.ps_priors[i_ps][i_ps_param].support)

                value = transform(states[i_param])

                log_density = transform.inv.log_abs_det_jacobian(value, states[i_param])
                log_density = sum_rightmost(log_density, log_density.dim() - value.dim() + self.ps_priors[i_ps][i_ps_param].event_dim)

                result[self.ps_param_labels[i_ps_param] + "_" + self.ps_labels[i_ps]] = pyro.sample(self.ps_param_labels[i_ps_param] + "_" + self.ps_labels[i_ps], dist.Delta(value, log_density=log_density, event_dim=self.ps_priors[i_ps][i_ps_param].event_dim))
                i_param += 1

        return result

    def init_mvn_guide(self):
        """ Initialize multivariate normal guide
        """
        init_loc = torch.full((self.n_params,), 0.0)
        init_scale = eye_like(init_loc, self.n_params) * 0.1

        _deep_setattr(self, "mvn.loc", PyroParam(init_loc, constraints.real))
        _deep_setattr(self, "mvn.scale_tril", PyroParam(init_scale, constraints.lower_cholesky))

    def init_iaf_guide(self, num_transforms=4, hidden_dims=None):
        """ Initialize IAF guide
        """
        num_transforms = 4
        if hidden_dims is None:
            hidden_dims = 2 * [10 * (self.n_params) + 1]
        context_dim = self.n_poiss + self.n_ps + 2

        self.prototype_tensor = torch.tensor(0.0)

        init_transform_fn = None
        if self.guide_name == "IAF":
            init_transform_fn = functools.partial(iterated, num_transforms, affine_autoregressive, hidden_dims=hidden_dims)
        elif self.guide_name == "ConditionalIAF":
            init_transform_fn = functools.partial(iterated, num_transforms, conditional_affine_autoregressive, hidden_dims=hidden_dims, context_dim=context_dim)

        self.transform = init_transform_fn(self.n_params)

        loc = self.prototype_tensor.new_zeros(1)
        scale = self.prototype_tensor.new_ones(1)
        self.base_dist = dist.Normal(loc, scale).expand([self.n_params]).to_event(1)

    def templates_guide_iaf(self, indices, gp_sample=None):
        """ IAF guide for template parameters
        """

        # Number of context variables (GP summary statistics) to condition IAF
        context_vars = torch.zeros(self.n_poiss + self.n_ps + 2)
        # context_vars = torch.zeros(2)

        td = None

        # IAF transformation either with or without conditioning on GP draw

        if self.guide_name == "IAF":

            td = dist.TransformedDistribution(self.base_dist, self.transform)

        elif self.guide_name == "ConditionalIAF":

            # Summary stats of GP draw---dor products of Poiss/non-Poiss templates with GP, as well as GP mean and variance
            context_vars[: self.n_poiss] = (self.poiss_temps[:, indices] @ gp_sample.exp().double()) / self.n_pix
            context_vars[self.n_poiss : self.n_poiss + self.n_ps] = (self.ps_temps[:, indices] @ gp_sample.exp().double()) / self.n_pix
            context_vars[-2] = torch.mean(gp_sample.exp())
            context_vars[-1] = torch.var(gp_sample.exp()).sqrt()

            td = dist.ConditionalTransformedDistribution(self.base_dist, self.transform).condition(context=context_vars)

        states = pyro.sample("states_" + self.name_prefix, td, infer={"is_auxiliary": True})

        result = {}

        for i_poiss in torch.arange(self.n_poiss):
            transform = biject_to(self.poiss_priors[i_poiss].support)
            value = transform(states[i_poiss])
            log_density = transform.inv.log_abs_det_jacobian(value, states[i_poiss])
            log_density = sum_rightmost(log_density, log_density.dim() - value.dim() + self.poiss_priors[i_poiss].event_dim)

            result[self.poiss_labels[i_poiss]] = pyro.sample(self.poiss_labels[i_poiss], dist.Delta(value, log_density=log_density, event_dim=self.poiss_priors[i_poiss].event_dim))

        i_param = self.n_poiss

        for i_ps in torch.arange(self.n_ps):
            for i_ps_param in torch.arange(self.n_ps_params):

                transform = biject_to(self.ps_priors[i_ps][i_ps_param].support)

                value = transform(states[i_param])

                log_density = transform.inv.log_abs_det_jacobian(value, states[i_param])
                log_density = sum_rightmost(log_density, log_density.dim() - value.dim() + self.ps_priors[i_ps][i_ps_param].event_dim)

                result[self.ps_param_labels[i_ps_param] + "_" + self.ps_labels[i_ps]] = pyro.sample(self.ps_param_labels[i_ps_param] + "_" + self.ps_labels[i_ps], dist.Delta(value, log_density=log_density, event_dim=self.ps_priors[i_ps][i_ps_param].event_dim))
                i_param += 1

        return result

    def get_exp_regions(self, nexp):

        # Determine the pixels of the exposure regions
        pix_array = np.where(self.mask == False)[0]
        exp_array = np.array([[pix_array[i], self.exposure_map[pix_array[i]]] for i in range(len(pix_array))])
        array_sorted = exp_array[np.argsort(exp_array[:, 1])]

        # Convert from list of exreg pixels to masks (int as used to index)
        array_split = np.array_split(array_sorted, nexp)
        expreg_array = [np.array([array_split[i][j][0] for j in range(len(array_split[i]))], dtype="int32") for i in range(len(array_split))]

        npix = len(self.mask)

        self.expreg_mask = []
        for i in range(nexp):
            temp_mask = np.logical_not(np.zeros(npix))
            for j in range(len(expreg_array[i])):
                temp_mask[expreg_array[i][j]] = False
            self.expreg_mask.append(temp_mask)

        # Store the total and region by region mean exposure
        expreg_values = [[array_split[i][j][1] for j in range(len(array_split[i]))] for i in range(len(array_split))]

        self.exposure_means_list = [np.mean(expreg_values[i]) for i in range(nexp)]
        self.exposure_mean = np.mean(self.exposure_means_list)

        self.expreg_indices = []
        for i in range(nexp):
            expreg_indices_temp = np.array([np.where(pix_array == elem)[0][0] for elem in expreg_array[i]])
            self.expreg_indices.append(torch.tensor(np.array(expreg_indices_temp)))


def _deep_setattr(obj, key, val):
    """
    Set an attribute `key` on the object. If any of the prefix attributes do
    not exist, they are set to :class:`~pyro.nn.PyroModule`.
    """

    def _getattr(obj, attr):
        obj_next = getattr(obj, attr, None)
        if obj_next is not None:
            return obj_next
        setattr(obj, attr, PyroModule())
        return getattr(obj, attr)

    lpart, _, rpart = key.rpartition(".")
    # Recursive getattr while setting any prefix attributes to PyroModule
    if lpart:
        obj = functools.reduce(_getattr, [obj] + lpart.split("."))
    setattr(obj, rpart, val)


def _deep_getattr(obj, key):
    for part in key.split("."):
        obj = getattr(obj, part)
    return obj

