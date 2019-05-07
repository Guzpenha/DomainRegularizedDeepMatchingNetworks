#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:turing:1
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

cd matchzoo


NUM_ITERS=100

#========================================#
#               DMN                      #
#========================================#

TRAINING_TYPE='None'

#--------------------------------#
#       Train on UDC             #
#--------------------------------#

srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters $NUM_ITERS --predict_ood False


#--------------------------------#
#       Train on Apple           #
#--------------------------------#

srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False 
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters $NUM_ITERS --predict_ood False

#--------------------------------#
#       Train on Both            #
#--------------------------------#

srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train -1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False 
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters $NUM_ITERS --predict_ood True


#========================================#
#               DMN-ADL                  #
#========================================#

TRAINING_TYPE='DMN-ADL'

#--------------------------------#
#       Train on UDC             #
#--------------------------------#

# udc -> apple
srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters 1$NUM_ITERS --predict_ood False --statistical_test 't-test'


#--------------------------------#
#       Train on Apple           #
#--------------------------------#

# apple -> udc
srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False 
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters 1$NUM_ITERS --predict_ood False --statistical_test 't-test'

#--------------------------------#
#       Train on Both            #
#--------------------------------#

srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train -1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False 
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters 1$NUM_ITERS --predict_ood True --statistical_test 't-test'


#========================================#
#               DMN-MTL                  #
#========================================#

TRAINING_TYPE='DMN-MTL'

#--------------------------------#
#       Train on UDC             #
#--------------------------------#

# udc -> apple
srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --input_to_domain_clf 'query_doc'
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters 2$NUM_ITERS --predict_ood False --statistical_test 't-test'


#--------------------------------#
#       Train on Apple           #
#--------------------------------#

# apple -> udc
srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False  --input_to_domain_clf 'query_doc'
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters 2$NUM_ITERS --predict_ood False --statistical_test 't-test'

#--------------------------------#
#       Train on Both            #
#--------------------------------#

srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train -1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False  --input_to_domain_clf 'query_doc'
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters 2$NUM_ITERS --predict_ood True --statistical_test 't-test'
