#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:turing:1
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

cd matchzoo


NUM_ITERS=100
CATEGORY='MSN'
#========================================#
#               DMN                      #
#========================================#

TRAINING_TYPE='None'
# srun python main_conversation_qa.py --phase train --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --domain_to_train 0
srun python main_conversation_qa.py --phase predict --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --test_weights_iters $NUM_ITERS --predict_ood False

#========================================#
#               DMN-ADL                  #
#========================================#

TRAINING_TYPE='DMN-ADL'
srun python main_conversation_qa.py --phase train --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --domain_to_train 0
srun python main_conversation_qa.py --phase predict --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --test_weights_iters 1$NUM_ITERS --predict_ood False

#========================================#
#               DMN-MTL                  #
#========================================#

TRAINING_TYPE='DMN-MTL'
srun python main_conversation_qa.py --phase train --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --domain_to_train 0
srun python main_conversation_qa.py --phase predict --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --test_weights_iters 2$NUM_ITERS --predict_ood False