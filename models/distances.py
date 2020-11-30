import gpytorch
import torch


class Distances:
    @staticmethod
    def default_postprocess_script(x):
        return x

    @classmethod
    def covar_dist_gcd(self, x1, x2, diag=False, last_dim_is_batch=False, dist_postprocess_func=default_postprocess_script, postprocess=True, lengthscale=1):
        r"""
        This is a helper method for computing the Euclidean distance between
        all pairs of points in x1 and x2.
        Args:
            :attr:`x1` (Tensor `n x d` or `b1 x ... x bk x n x d`):
                First set of data.
            :attr:`x2` (Tensor `m x d` or `b1 x ... x bk x m x d`):
                Second set of data.
            :attr:`diag` (bool):
                Should we return the whole distance matrix, or just the diagonal? If True, we must have `x1 == x2`.
            :attr:`last_dim_is_batch` (tuple, optional):
                Is the last dimension of the data a batch dimension or not?
            :attr:`square_dist` (bool):
                Should we square the distance matrix before returning?
        Returns:
            (:class:`Tensor`, :class:`Tensor) corresponding to the distance matrix between `x1` and `x2`.
            The shape depends on the kernel's mode
            * `diag=False`
            * `diag=False` and `last_dim_is_batch=True`: (`b x d x n x n`)
            * `diag=True`
            * `diag=True` and `last_dim_is_batch=True`: (`b x d x n`)
        """
        if last_dim_is_batch:
            x1 = x1.transpose(-1, -2).unsqueeze(-1)
            x2 = x2.transpose(-1, -2).unsqueeze(-1)

        x1_eq_x2 = torch.equal(x1, x2)

        # torch scripts expect tensors
        postprocess = torch.tensor(postprocess)

        res = None

        if diag:
            # Special case the diagonal because we can return all zeros most of the time.
            if x1_eq_x2:
                res = torch.zeros(*x1.shape[:-2], x1.shape[-2], dtype=x1.dtype, device=x1.device) / lengthscale
                if postprocess:
                    res = dist_postprocess_func(res)
                return res
            else:
                res = self.gcd(x1, x2) / lengthscale
            if postprocess:
                res = dist_postprocess_func(res)
            return res
        else:
            res = self.gcd_p(x1, x2) / lengthscale
            if postprocess:
                res = dist_postprocess_func(res)
        return res

    @staticmethod
    def gcd(ang1, ang2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        lon1, lat1 = ang1.T
        lon2, lat2 = ang2.T
        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
        return 2 * torch.asin(torch.sqrt(a))

    @staticmethod
    def gcd_p(ang1, ang2):
        """
        Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
        """
        x_1, x_2 = ang1, ang2

        x_1_u = torch.unsqueeze(x_1, -2)
        x_2_u = torch.unsqueeze(x_2, -3)

        d = torch.sin((x_2_u - x_1_u) / 2) ** 2

        lat1 = torch.unsqueeze(x_1[:, 1], -1)
        lat2 = torch.unsqueeze(x_2[:, 1], -2)

        cos_prod = torch.cos(lat1) * torch.cos(lat2)

        a = d[:, :, 0] * cos_prod + d[:, :, 1]

        return 2 * torch.asin(torch.sqrt(a))
