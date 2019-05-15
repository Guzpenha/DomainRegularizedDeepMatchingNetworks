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


NUM_ITERS=100
RESET_CLF_WEIGHTS=1000 #no resetting
CATEGORY='MSN'
for RANDOM_SEED in 10 100 1000
do
	for LAMBDA in 0.01 0.06309573444801933 0.15848931924611143 0.3981071705534973 1.0
	do
		for INPUT_TO_CLF in 'query_doc' 'match'
		do
			#========================================#
			#               DMN-ADL                  #
			#========================================#

			TRAINING_TYPE='DMN-ADL'
			srun python main_conversation_qa.py --phase train --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0 --reset_clf_weights_iters $RESET_CLF_WEIGHTS --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED --l $LAMBDA
			srun python main_conversation_qa.py --phase predict --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --test_weights_iters 1${NUM_ITERS}11111 --predict_ood False --statistical_test 't-test'

			#========================================#
			#               DMN-MTL                  #
			#========================================#

			TRAINING_TYPE='DMN-MTL'
			srun python main_conversation_qa.py --phase train --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0 --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED --l $LAMBDA
			srun python main_conversation_qa.py --phase predict --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --test_weights_iters 2${NUM_ITERS}11111 --predict_ood False --statistical_test 't-test'
		done
	done
done