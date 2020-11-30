import gpytorch
import torch

from models.distances import Distances


class CustomRBFKernel(gpytorch.kernels.Kernel):

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):
        covar = Distances.covar_dist_gcd(x1, x2, diag=diag, dist_postprocess_func=self.postprocess_rbf, postprocess=True, lengthscale=self.lengthscale, **params)

        if diag:
            return covar[0]
        return covar

    def postprocess_rbf(self, dist_mat):
        return dist_mat.pow(2).div_(-2).exp_()

class CustomMatern12Kernel(gpytorch.kernels.Kernel):

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):

        covar = Distances.covar_dist_gcd(x1, x2, diag=diag, dist_postprocess_func=self.postprocess_matern12, postprocess=True, lengthscale=self.lengthscale, **params)

        if diag:
            return covar[0]
        return covar

    def postprocess_matern12(self, dist_mat):
        return (-dist_mat).exp_()

class CustomMatern32Kernel(gpytorch.kernels.Kernel):

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):

        distance = Distances.covar_dist_gcd(x1, x2, diag=diag, postprocess=False, lengthscale=self.lengthscale, **params)
        exp_component = (-torch.sqrt(torch.tensor(3.0)) * distance).exp_()
        constant_component = (torch.sqrt(torch.tensor(3.0)) * distance).add(1)
        covar = constant_component * exp_component

        if diag:
            return covar[0]
        return covar

class CustomMatern52Kernel(gpytorch.kernels.Kernel):

    has_lengthscale = True

    def forward(self, x1, x2, diag=False, **params):

        distance = Distances.covar_dist_gcd(x1, x2, diag=diag, postprocess=False, lengthscale=self.lengthscale, **params)
        exp_component = (-torch.sqrt(torch.tensor(5.0)) * distance).exp_()
        constant_component = (torch.sqrt(torch.tensor(5.0)) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
        covar = constant_component * exp_component

        if diag:
            return covar[0]
        return covar