import argparse

from inference.constructor import ModelConstructor
from inference.trainer import PyroSVITrainer

parser = argparse.ArgumentParser()
parser.add_argument("--subsample_size", action="store", dest="subsample_size", default=1500, type=int)
parser.add_argument("--num_iter", action="store", dest="num_iter", default=50000, type=int)
parser.add_argument("--poiss_only", action="store_true", dest="poiss_only")
parser.add_argument("--guide_name", action="store", dest="guide_name", default="MVN", type=str)
parser.add_argument("--run_prefix", action="store", dest="run_prefix", default="test", type=str)
parser.add_argument("--outputscale_prior", nargs="+", action="store", dest="outputscale_prior", default=[0.5, 0.01], type=float)
parser.add_argument("--lengthscale_prior", nargs="+", action="store", dest="lengthscale_prior", default=[0.2, 0.001], type=float)
parser.add_argument("--r_outer", action="store", dest="r_outer", default=20.0, type=float)
parser.add_argument("--num_inducing", action="store", dest="num_inducing", default=200, type=int)

results = parser.parse_args()

subsample_size = results.subsample_size
num_iter = results.num_iter
poiss_only = results.poiss_only
guide_name = results.guide_name
run_prefix = results.run_prefix
lengthscale_prior = results.lengthscale_prior
outputscale_prior = results.outputscale_prior
r_outer = results.r_outer
num_inducing = results.num_inducing

run_name = "%s_inducing_%s_r_%s_guide_%s_ss_%s_os_%s_ls_%s_poiss_%s" % (run_prefix, str(num_inducing), str(r_outer), guide_name, str(subsample_size), str(outputscale_prior), str(lengthscale_prior), str(poiss_only))

# Hard-coded Poissonian norms and non-Poissonian parameters for experiment in paper
theta_poiss = [0.5, 0.1, 0.5, 0.9, 8.0, 4.0]

# If Poisson only, don't simulate any PS contribution
if poiss_only:
    theta_ps = [0.0, 20.0, 1.8, -20.0, 20.0, 0.1]
else:
    theta_ps = [1.5, 20.0, 1.8, -20.0, 20.0, 0.1]

# Instantiate model
mc = ModelConstructor(nside=128, dif_sim="mO", dif_fit="p6", theta_poiss=theta_poiss, theta_ps=theta_ps, guide_name=guide_name, num_inducing=num_inducing, kernel="matern52", run_name=run_name, r_outer=r_outer, lengthscale_prior=lengthscale_prior, outputscale_prior=outputscale_prior, inducing_strategy="uniform", learn_inducing_locations=True, poiss_only=poiss_only)

# If subsample size is 0, use all pixels (no subsampling)
if subsample_size > 0:
    batch_size = subsample_size
else:
    batch_size = len(mc.train_x)

# Train and save
trainer = PyroSVITrainer(mc.model, mc.train_x, mc.train_y, mc.save_dir)
trainer.train(num_iter=num_iter, batch_size=batch_size, decay_every_steps=1000, save_every_steps=1000)
