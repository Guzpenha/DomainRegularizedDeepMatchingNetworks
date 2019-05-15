#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

cd matchzoo


NUM_ITERS=50
RESET_CLF_WEIGHTS=1000 #no resetting
for RANDOM_SEED in 10 100 1000
do
	for LAMBDA in 0.01 0.06309573444801933 0.15848931924611143 0.3981071705534973 1.0
	do
		#========================================#
		#               DMN-ADL                  #
		#========================================#

		TRAINING_TYPE='DMN-ADL'

		for INPUT_TO_CLF in 'query_doc' 'match'
		do
			#--------------------------------#
			#       Train on UDC             #
			#--------------------------------#


			srun python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --reset_clf_weights_iters $RESET_CLF_WEIGHTS --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED --lambda $LAMBDA
			srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 1${NUM_ITERS}22222 --predict_ood False --statistical_test 't-test'
			 
		done

		#========================================#
		#               DMN-MTL                  #
		#========================================#

		TRAINING_TYPE='DMN-MTL'

		for INPUT_TO_CLF in 'query_doc' 'match'
		do

			#--------------------------------#
			#       Train on UDC             #
			#--------------------------------#

			srun python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --input_to_domain_clf $INPUT_TO_CLF  --keras_random_seed $RANDOM_SEED --lambda $LAMBDA
			srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 2${NUM_ITERS}22222 --predict_ood False --statistical_test 't-test'
		done
	done
done