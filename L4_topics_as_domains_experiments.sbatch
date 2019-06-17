#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=100:00:00
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
	for CATEGORY in 'Home & Garden' 'Sports' 'Politics & Government' 'Travel' #'Pregnancy & Parenting' 'Travel' 'Science & Mathematics' 'Home & Garden' 'Education & Reference' 'Business & Finance' 'Food & Drink' 'Cars & Transportation' 'Politics & Government' 'Games & Recreation' 'Society & Culture' 'Family & Relationships' 'Entertainment & Music' 'Sports' 'Consumer Electronics' 'Health' 'Pets' 'News & Events' 'Arts & Humanities' 'Dining Out' 'Social Science' 'Beauty & Style' 'undefined' 'Local Businesses' 'Environment' 'Asia Pacific' 'Societ\xc3\xa0 e culture'
	do
		#========================================#
		#               DMN                      #
		#========================================#

		TRAINING_TYPE='None'
		srun python main_conversation_qa.py --phase train --model_file config/L4/dmn_cnn_topics_as_domains.config --or_cmd True --test_category "${CATEGORY}" --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0 --keras_random_seed $RANDOM_SEED
		srun python main_conversation_qa.py --phase predict --model_file config/L4/dmn_cnn_topics_as_domains.config --or_cmd True --test_category "${CATEGORY}" --test_weights_iters ${NUM_ITERS} --predict_ood False

		for INPUT_TO_CLF in 'query_doc' 'match'
		do
			#========================================#
			#               DMN-ADL                  #
			#========================================#

			TRAINING_TYPE='DMN-ADL'
			srun python main_conversation_qa.py --phase train --model_file config/L4/dmn_cnn_topics_as_domains.config --or_cmd True --test_category "${CATEGORY}" --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0 --reset_clf_weights_iters $RESET_CLF_WEIGHTS --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED
			srun python main_conversation_qa.py --phase predict --model_file config/L4/dmn_cnn_topics_as_domains.config --or_cmd True --test_category "${CATEGORY}" --test_weights_iters 1${NUM_ITERS}11111 --predict_ood False

			#========================================#
			#               DMN-MTL                  #
			#========================================#

			TRAINING_TYPE='DMN-MTL'
			srun python main_conversation_qa.py --phase train --model_file config/L4/dmn_cnn_topics_as_domains.config --or_cmd True --test_category "${CATEGORY}" --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0 --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED
			srun python main_conversation_qa.py --phase predict --model_file config/L4/dmn_cnn_topics_as_domains.config --or_cmd True --test_category "${CATEGORY}" --test_weights_iters 2${NUM_ITERS}11111 --predict_ood False
		done
	done
done
