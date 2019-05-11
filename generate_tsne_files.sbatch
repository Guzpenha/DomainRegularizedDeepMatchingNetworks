#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=40000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

cd matchzoo

NUM_ITERS=100
BASE_DIR=/tudelft.net/staff-umbrella/domaincss/NeuralResponseRankingDAL/
DOMAIN_1=ms
DOMAIN_2=udc

## MATCHING REP for ${DOMAIN_1}_${DOMAIN_2}

TRAINING_TYPE='None'
#random weights
srun python main_conversation_qa.py --phase predict --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --random_weights_predict True --predict_ood False --save_query_representation match
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt_dmn_random_match_rep
cp ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/q_rep.pickle ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/q_rep.pickle_dmn_random_match_rep
srun python breakdown_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_random match_rep
srun python classifier_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_random match_rep
# trained only on ${DOMAIN_1}
srun python main_conversation_qa.py --phase predict --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --test_weights_iters ${NUM_ITERS}11111 --predict_ood False --save_query_representation match
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt_dmn_${DOMAIN_1}_match_rep
cp ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/q_rep.pickle ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/q_rep.pickle_dmn_${DOMAIN_1}_match_rep
srun python breakdown_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1} match_rep
srun python classifier_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1} match_rep
# trained on both
srun python main_conversation_qa.py --phase predict --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --test_weights_iters ${NUM_ITERS}00000 --predict_ood False --save_query_representation match
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt_dmn_both_match_rep
cp ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/q_rep.pickle ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/q_rep.pickle_dmn_both_match_rep
srun python breakdown_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_both match_rep
srun python classifier_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_both match_rep

TRAINING_TYPE='DMN-ADL'
srun python main_conversation_qa.py --phase predict --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --test_weights_iters 1${NUM_ITERS}11111 --predict_ood False --save_query_representation match
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt_ADL_trained_on_domain_1 ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt_ADL_trained_on_domain_1 > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt_ADL_trained_on_domain_1 ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt_ADL_trained_on_domain_1 > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt_dmn_adl_match_rep
cp ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/q_rep.pickle ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/q_rep.pickle_dmn_adl_match_rep
srun python breakdown_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_adl match_rep
srun python classifier_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_adl match_rep

# TEXTUAL REP for ${DOMAIN_1}_${DOMAIN_2}

TRAINING_TYPE='None'
#random weights
srun python main_conversation_qa.py --phase train --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --predict False
srun python main_conversation_qa.py --phase predict --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --random_weights_predict True --predict_ood False --save_query_representation text
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
srun python breakdown_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_random_textual turn_1
srun python classifier_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_random_textual turn_1
# trained only on ${DOMAIN_1}
srun python main_conversation_qa.py --phase predict --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --test_weights_iters ${NUM_ITERS}11111 --predict_ood False --save_query_representation text
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
srun python breakdown_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1}_textual turn_1
srun python classifier_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1}_textual turn_1
# trained on both
srun python main_conversation_qa.py --phase predict --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --test_weights_iters ${NUM_ITERS}00000 --predict_ood False --save_query_representation text
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
srun python breakdown_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_both_textual turn_1
srun python classifier_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_both_textual turn_1

TRAINING_TYPE='DMN-MTL'
srun python main_conversation_qa.py --phase predict --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --test_weights_iters 2${NUM_ITERS}11111 --predict_ood False --save_query_representation text
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt_MTL_trained_on_domain_1 ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt_MTL_trained_on_domain_1 > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
srun python breakdown_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_mtl_textual turn_1
srun python classifier_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_mtl_textual turn_1

# SENTENCE REP for ${DOMAIN_1}_${DOMAIN_2}

TRAINING_TYPE='None'
#random weights
srun python main_conversation_qa.py --phase train --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --predict False
srun python main_conversation_qa.py --phase predict --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --random_weights_predict True --predict_ood False --save_query_representation sentence
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
srun python breakdown_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_random_sentence turn_1
srun python classifier_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_random_sentence turn_1
# trained only on ${DOMAIN_1}
srun python main_conversation_qa.py --phase predict --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --test_weights_iters ${NUM_ITERS}11111 --predict_ood False --save_query_representation sentence
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
srun python breakdown_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1}_sentence turn_1
srun python classifier_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_${DOMAIN_1}_sentence turn_1
# trained on both
srun python main_conversation_qa.py --phase predict --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --test_weights_iters ${NUM_ITERS}00000 --predict_ood False --save_query_representation sentence
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
srun python breakdown_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_both_sentence turn_1
srun python classifier_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_both_sentence turn_1

TRAINING_TYPE='DMN-MTL'
srun python main_conversation_qa.py --phase predict --model_file config/${DOMAIN_1}_${DOMAIN_2}/dmn_cnn_${DOMAIN_1}_${DOMAIN_2}.config --or_cmd True --test_weights_iters 2${NUM_ITERS}11111 --predict_ood False --save_query_representation sentence
cat ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}.test.txt_MTL_trained_on_domain_1 ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_2}.test.txt_MTL_trained_on_domain_1 > ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt
srun python classifier_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_mtl_sentence turn_1
srun python breakdown_analyses.py ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelRes/dmn_cnn.predict_${DOMAIN_1}_and_${DOMAIN_2}.test.txt ${BASE_DIR}data/${DOMAIN_1}_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_2}/ModelInput/dmn_model_input/ ${BASE_DIR}data/${DOMAIN_1}_${DOMAIN_2}/ModelInput/dmn_model_input/ _dmn_mtl_sentence turn_1