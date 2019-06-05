#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=70:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

cd matchzoo


INPUT_TO_CLF='query_doc'
CATEGORY='MSN'
NUM_ITERS=100
RESET_CLF_WEIGHTS=1000 #no resetting

for LAMBDA in 0.00 0.05 0.10 0.15 0.20 0.25 0.30 0.35 #0.40 0.45 0.50 0.55 0.60 0.65 0.70 0.75 0.80 0.85 0.90 0.95 1.0
do
	for RANDOM_SEED in 10 100 1000 10000 100000 11 101 1001 10001 100001
	do
		#========================================#
		#               DMN-ADL                  #
		#========================================#

		TRAINING_TYPE='DMN-ADL'
		srun python main_conversation_qa.py --phase train --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0 --reset_clf_weights_iters $RESET_CLF_WEIGHTS --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED --l $LAMBDA
		srun python main_conversation_qa.py --phase predict --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --test_weights_iters 1${NUM_ITERS}11111 --predict_ood False

		#========================================#
		#               DMN-MTL                  #
		#========================================#

		TRAINING_TYPE='DMN-MTL'
		srun python main_conversation_qa.py --phase train --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0 --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED --l $LAMBDA
		srun python main_conversation_qa.py --phase predict --model_file config/ms_v2/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --test_weights_iters 2${NUM_ITERS}11111 --predict_ood False
	done
done