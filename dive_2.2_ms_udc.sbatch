#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=08:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

cd matchzoo

NUM_ITERS=100
RESET_CLF_WEIGHTS=1000

#========================================#
#               DMN-MTL                  #
#========================================#

TRAINING_TYPE='DMN-MTL'

for INPUT_TO_CLF in 'match'
do

	#--------------------------------#
	#       Train on MSDialog        #
	#--------------------------------#
	srun python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False  --input_to_domain_clf $INPUT_TO_CLF
	srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 2${NUM_ITERS}11111 --predict_ood False --statistical_test 't-test'

	#--------------------------------#
	#       Train on UDC             #
	#--------------------------------#

	srun python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --input_to_domain_clf $INPUT_TO_CLF 
	srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 2${NUM_ITERS}22222 --predict_ood False --statistical_test 't-test'
done
