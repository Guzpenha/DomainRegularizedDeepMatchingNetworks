#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:turing:1
#SBATCH --mem-per-cpu=10000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

cd matchzoo

NUM_ITERS=100
BASE_DIR=/tudelft.net/staff-umbrella/domaincss/NeuralResponseRankingDAL/
## MATCHING REP for ms_apple

TRAINING_TYPE='None'
# trained only on ms
srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters ${NUM_ITERS}11111 --predict_ood False --save_query_representation match
cat ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms.test.txt ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_apple.test.txt > ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms_and_apple.test.txt
srun python breakdown_analyses.py ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms_and_apple.test.txt ${BASE_DIR}data/ms_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/apple/ModelInput/dmn_model_in${BASE_DIR}ms_apple/ModelInput/dmn_model_input/ _dmn_ms match_rep
# trained on both
srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters ${NUM_ITERS}00000 --predict_ood False --save_query_representation match
cat ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms.test.txt ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_apple.test.txt > ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms_and_apple.test.txt
srun python breakdown_analyses.py ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms_and_apple.test.txt ${BASE_DIR}data/ms_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/apple/ModelInput/dmn_model_input/ ${BASE_DIR}data/ms_apple/ModelInput/dmn_model_input/ _dmn_both match_rep

TRAINING_TYPE='DMN-ADL'
srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 1${NUM_ITERS}11111 --predict_ood False --save_query_representation match
cat ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms.test.txt_ADL_trained_on_domain_1 ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_apple.test.txt_ADL_trained_on_domain_1 > ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms_and_apple.test.txt
srun python breakdown_analyses.py ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms_and_apple.test.txt ${BASE_DIR}data/ms_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/apple/ModelInput/dmn_model_input/ ${BASE_DIR}data/ms_apple/ModelInput/dmn_model_input/ _dmn_adl match_rep

## TEXTUAL REP for ms_apple

TRAINING_TYPE='None'
# trained only on ms
srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters ${NUM_ITERS}11111 --predict_ood False --save_query_representation text
cat ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms.test.txt ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_apple.test.txt > ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms_and_apple.test.txt
srun python breakdown_analyses.py ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms_and_apple.test.txt ${BASE_DIR}data/ms_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/apple/ModelInput/dmn_model_input/ ${BASE_DIR}data/ms_apple/ModelInput/dmn_model_input/ _dmn_ms_textual turn_1
# trained on both
srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters ${NUM_ITERS}00000 --predict_ood False --save_query_representation text
cat ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms.test.txt ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_apple.test.txt > ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms_and_apple.test.txt
srun python breakdown_analyses.py ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms_and_apple.test.txt ${BASE_DIR}data/ms_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/apple/ModelInput/dmn_model_input/ ${BASE_DIR}data/ms_apple/ModelInput/dmn_model_input/ _dmn_both_textual turn_1

TRAINING_TYPE='DMN-MTL'
srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 2${NUM_ITERS}11111 --predict_ood False --save_query_representation text
cat ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms.test.txt_MTL_trained_on_domain_1 ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_apple.test.txt_MTL_trained_on_domain_1 > ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms_and_apple.test.txt
srun python breakdown_analyses.py ${BASE_DIR}data/ms_apple/ModelRes/dmn_cnn.predict_ms_and_apple.test.txt ${BASE_DIR}data/ms_v2/ModelInput/dmn_model_input/ ${BASE_DIR}data/apple/ModelInput/dmn_model_input/ ${BASE_DIR}data/ms_apple/ModelInput/dmn_model_input/ _dmn_mtl_textual turn_1
