#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

cd matchzoo

srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --or_cmd True --random_weights_predict True --predict_ood False
srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --or_cmd True --random_weights_predict True --predict_ood False
srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --or_cmd True --random_weights_predict True --predict_ood False
srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --or_cmd True --random_weights_predict True --predict_ood False
srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --or_cmd True --random_weights_predict True --predict_ood False
