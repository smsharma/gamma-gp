import os

batch = """#!/bin/bash
#SBATCH -N 1   # node count
#SBATCH --ntasks-per-node=1
#SBATCH -t 10:00:00
#SBATCH --mem=4GB
#SBATCH --mail-type=begin
#SBATCH --mail-type=end
#SBATCH --mail-user=sm8383@nyu.edu
cd /home/sm8383/fermi-gce-gp/
conda activate
"""

guide_names = ["MVN", "IAF", "ConditionalIAF"]
num_inducing_ary = [200]

for guide_name in guide_names:
    for num_inducing in num_inducing_ary:
        batchn = batch + "\n"
        batchn += "python train.py --guide_name " + guide_name + " --num_inducing " + str(num_inducing) + " --poiss_only"
        fname = "batch/submit.batch"
        f = open(fname, "w")
        f.write(batchn)
        f.close()
        os.system("chmod +x " + fname)
        os.system("sbatch " + fname)
