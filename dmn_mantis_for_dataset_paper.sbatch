#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:turing:1
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

cd matchzoo


NUM_ITERS=100

#========================================#
#               DMN                      #
#========================================#
for RANDOM_SEED in 10 100 1000 10000 100000
do
	srun python main_conversation_qa.py --phase train --model_file config/mantis/dmn_cnn.config --or_cmd True --num_iters ${NUM_ITERS} --predict False --keras_random_seed $RANDOM_SEED
	srun python main_conversation_qa.py --phase predict --model_file config/mantis/dmn_cnn.config --or_cmd True --test_weights_iters ${NUM_ITERS} --predict_ood True
done