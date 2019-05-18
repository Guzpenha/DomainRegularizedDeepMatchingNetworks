#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=20000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

cd matchzoo

BASE_DIR=/tudelft.net/staff-umbrella/domaincss/NeuralResponseRankingDAL/
NUM_ITERS=100
RESET_CLF_WEIGHTS=1000
CATEGORY='askubuntu'
DOMAIN_1=in
DOMAIN_2=out

#random weights
# srun python main_conversation_qa.py --phase predict --model_file config/mantis/dmn_cnn_topics_as_domains.config --or_cmd True --random_weights_predict True --predict_ood False --save_query_representation match
# cat ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
# cat ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt_dmn_random_match_rep
# cp ${BASE_DIR}data/mantis/ModelRes/q_rep.pickle ${BASE_DIR}data/mantis/ModelRes/q_rep.pickle_dmn_random_match_rep
# srun python breakdown_analyses.py ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/mantis/ModelInput/dmn_model_input/ _dmn_random match_rep
# srun python classifier_analyses.py ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/mantis/ModelInput/dmn_model_input/ _dmn_random match_rep

TRAINING_TYPE='None'
# srun python main_conversation_qa.py --phase train --model_file config/mantis/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0
srun python main_conversation_qa.py --phase predict --model_file config/mantis/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --test_weights_iters ${NUM_ITERS} --predict_ood False --save_query_representation text

cat ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
cat ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt_dmn_${DOMAIN_1}_textual_rep
cp ${BASE_DIR}data/mantis/ModelRes/q_rep.pickle ${BASE_DIR}data/mantis/ModelRes/q_rep.pickle_dmn_${DOMAIN_1}_textual_rep
srun python breakdown_analyses.py ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/mantis/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1}_textual  turn_1
srun python classifier_analyses.py ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/mantis/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1}_textual  turn_1

TRAINING_TYPE='None'
# srun python main_conversation_qa.py --phase train --model_file config/mantis/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0
srun python main_conversation_qa.py --phase predict --model_file config/mantis/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --test_weights_iters ${NUM_ITERS} --predict_ood False --save_query_representation sentence

cat ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
cat ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt_dmn_${DOMAIN_1}_sentence_rep
cp ${BASE_DIR}data/mantis/ModelRes/q_rep.pickle ${BASE_DIR}data/mantis/ModelRes/q_rep.pickle_dmn_${DOMAIN_1}_textual_rep
srun python breakdown_analyses.py ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/mantis/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1}_sentence  turn_1
srun python classifier_analyses.py ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/mantis/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1}_sentence  turn_1


TRAINING_TYPE='DMN-MTL'
INPUT_TO_CLF='query_doc'
# srun python main_conversation_qa.py --phase train --model_file config/mantis/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0 --reset_clf_weights_iters $RESET_CLF_WEIGHTS --input_to_domain_clf $INPUT_TO_CLF
srun python main_conversation_qa.py --phase predict --model_file config/mantis/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --test_weights_iters 2${NUM_ITERS}11111 --predict_ood False --save_query_representation text

cat ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
cat ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt_dmn_mtl_textual_rep
cp ${BASE_DIR}data/mantis/ModelRes/q_rep.pickle ${BASE_DIR}data/mantis/ModelRes/q_rep.pickle_dmn_mtl_match_rep
srun python breakdown_analyses.py ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/mantis/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1}_mtl_textual  turn_1
srun python classifier_analyses.py ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/mantis/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1}_mtl_textual  turn_1

srun python main_conversation_qa.py --phase predict --model_file config/mantis/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --test_weights_iters 2${NUM_ITERS}11111 --predict_ood False --save_query_representation sentence

cat ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
cat ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt_dmn_mtl_sentence_rep
cp ${BASE_DIR}data/mantis/ModelRes/q_rep.pickle ${BASE_DIR}data/mantis/ModelRes/q_rep.pickle_dmn_mtl_match_rep
srun python breakdown_analyses.py ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/mantis/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1}_mtl_sentence  turn_1
srun python classifier_analyses.py ${BASE_DIR}data/mantis/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/mantis/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1}_mtl_sentence  turn_1

# for INPUT_TO_CLF in 'query_doc' 'match'
# do
# 	#========================================#
# 	#               DMN-ADL                  #
# 	#========================================#

# 	TRAINING_TYPE='DMN-ADL'
# 	srun python main_conversation_qa.py --phase train --model_file config/mantis/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0 --reset_clf_weights_iters $RESET_CLF_WEIGHTS --input_to_domain_clf $INPUT_TO_CLF
# 	srun python main_conversation_qa.py --phase predict --model_file config/mantis/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --test_weights_iters 1${NUM_ITERS}11111 --predict_ood False --statistical_test 't-test'

# 	#========================================#
# 	#               DMN-MTL                  #
# 	#========================================#

# 	TRAINING_TYPE='DMN-MTL'
# 	srun python main_conversation_qa.py --phase train --model_file config/mantis/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0 --input_to_domain_clf $INPUT_TO_CLF
# 	srun python main_conversation_qa.py --phase predict --model_file config/mantis/dmn_cnn_topics_as_domains.config --or_cmd True --test_category $CATEGORY --test_weights_iters 2${NUM_ITERS}11111 --predict_ood False --statistical_test 't-test'
# done