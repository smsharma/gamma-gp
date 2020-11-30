import numpy as np
import pyro
import torch
import gpytorch
from tqdm.autonotebook import tqdm


class PyroSVITrainer:
    """ Train Pyro model
    """

    def __init__(self, model, train_x, train_y, save_dir=None, optimizer=torch.optim.Adam, optimizer_kwargs={"lr": 2e-3, "betas": (0.9, 0.999)}, scheduler=pyro.optim.ExponentialLR, scheduler_kwargs={"gamma": 0.95}, init_lengthscale=0.2):

        pyro.clear_param_store()

        self.save_dir = save_dir

        self.model = model
        self.train_x = train_x
        self.train_y = train_y

        scheduler_kwargs = {"optimizer": optimizer, "optim_args": optimizer_kwargs, **scheduler_kwargs}
        self.scheduler = scheduler(scheduler_kwargs)

        self.model.covar_module.base_kernel.lengthscale = init_lengthscale
        self.loss = []

    def train(self, num_iter, batch_size, decay_every_steps, save_every_steps, vectorize_particles=False, num_particles=1):
        """ SVI training loop
        """

        elbo = pyro.infer.Trace_ELBO(retain_graph=True, vectorize_particles=vectorize_particles, num_particles=num_particles)
        svi = pyro.infer.SVI(model=self.model.model, guide=self.model.guide, optim=self.scheduler, loss=elbo)

        self.model.train()
        iterator = tqdm(range(num_iter))

        for i in iterator:

            indices = torch.randperm(self.train_x.size()[0])[:batch_size]

            self.model.zero_grad()

            loss = svi.step(self.train_x[indices], self.train_y[indices], indices)

            if i % decay_every_steps == 1:
                self.scheduler.step()

            if (i % save_every_steps == 1) and (self.save_dir is not None):
                self.save_checkpoint(i)

            self.loss.append(loss / batch_size)
            iterator.set_postfix(loss=loss / batch_size, lengthscale=self.model.covar_module.base_kernel.lengthscale.item())

    def train_natural(self, num_iter, batch_size, save_every_steps):
        """ Train GP variational parameters with natural gradient descent. Doesn't work well at the moment...
        """

        elbo = pyro.infer.Trace_ELBO(retain_graph=True)

        variational_ngd_optimizer = gpytorch.optim.NGD(self.model.variational_parameters(), num_data=batch_size, lr=1e-6)
        hyperparameter_optimizer = torch.optim.Adam(self.model.hyperparameters(), lr=4e-3)

        self.model.train()
        iterator = tqdm(range(num_iter))

        for i in iterator:

            self.model.zero_grad()

            variational_ngd_optimizer.zero_grad()
            hyperparameter_optimizer.zero_grad()

            indices = torch.randperm(self.train_x.size()[0])[:batch_size]

            loss = elbo.differentiable_loss(self.model.model, self.model.guide, self.train_x[indices], self.train_y[indices], indices)
            loss.backward(retain_graph=True)

            variational_ngd_optimizer.step()
            hyperparameter_optimizer.step()

            if (i % save_every_steps == 1) and (self.save_dir is not None):
                self.save_checkpoint(i)

            self.loss.append(loss / batch_size)
            iterator.set_postfix(loss=loss / batch_size, lengthscale=self.model.covar_module.base_kernel.lengthscale.item())

    def save_checkpoint(self, i):
        """ Save state dict of model and optimizer at a given iteration `i`
        """
        torch.save({"model_state_dict": self.model.state_dict(), "loss": torch.tensor(self.loss)}, self.save_dir + "/model_%s.pt" % (str(i)))

        self.scheduler.save(self.save_dir + "/opt_%s.pt" % (str(i)))

    def load_checkpoint(self, i):
        """ Load state dict of model and optimizer at a given iteration `i`
        """
        checkpoint = torch.load(self.save_dir + "/model_%s.pt" % (str(i)))
        self.loss = checkpoint["loss"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.scheduler.load(self.save_dir + "/opt_%s.pt" % (str(i)))
