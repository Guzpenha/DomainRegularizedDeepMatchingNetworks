#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=15:00:00
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


for RANDOM_SEED in 10 100 1000 10000 100000
do
	#========================================#
	#               DMN                      #
	#========================================#

	TRAINING_TYPE='None'

	#--------------------------------#
	#       Train on MSDialog        #
	#--------------------------------#

	srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False  --keras_random_seed $RANDOM_SEED
	srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters ${NUM_ITERS}11111 --predict_ood False


	#--------------------------------#
	#       Train on Apple           #
	#--------------------------------#

	srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False  --keras_random_seed $RANDOM_SEED
	srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters ${NUM_ITERS}22222 --predict_ood False


	#--------------------------------#
	#       Train on Both            #
	#--------------------------------#

	srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train -1 --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --keras_random_seed $RANDOM_SEED
	srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters ${NUM_ITERS}00000 --predict_ood True


	#========================================#
	#               DMN-ADL                  #
	#========================================#

	TRAINING_TYPE='DMN-ADL'

	for INPUT_TO_CLF in 'query_doc' 'match'
	do
		#--------------------------------#
		#       Train on MSDialog        #
		#--------------------------------#

		srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --reset_clf_weights_iters $RESET_CLF_WEIGHTS --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED
		srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 1${NUM_ITERS}11111 --predict_ood False --statistical_test 't-test'


		#--------------------------------#
		#       Train on Apple           #
		#--------------------------------#

		srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --reset_clf_weights_iters $RESET_CLF_WEIGHTS --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED
		srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 1${NUM_ITERS}22222 --predict_ood False --statistical_test 't-test'

		#--------------------------------#
		#       Train on Both            #
		#--------------------------------#

		srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train -1 --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False  --reset_clf_weights_iters $RESET_CLF_WEIGHTS --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED
		srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 1${NUM_ITERS}00000 --predict_ood True --statistical_test 't-test'
	done

	#========================================#
	#               DMN-MTL                  #
	#========================================#

	TRAINING_TYPE='DMN-MTL'

	for INPUT_TO_CLF in 'query_doc' 'match'
	do

		#--------------------------------#
		#       Train on MSDialog        #
		#--------------------------------#

		srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False  --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED
		srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 2${NUM_ITERS}11111 --predict_ood False --statistical_test 't-test'


		#--------------------------------#
		#       Train on Apple           #
		#--------------------------------#

		srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False  --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED
		srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 2${NUM_ITERS}22222 --predict_ood False --statistical_test 't-test'

		#--------------------------------#
		#       Train on Both            #
		#--------------------------------#

		srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train -1 --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False  --input_to_domain_clf $INPUT_TO_CLF  --keras_random_seed $RANDOM_SEED
		srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 2${NUM_ITERS}00000 --predict_ood True --statistical_test 't-test'

	done
done