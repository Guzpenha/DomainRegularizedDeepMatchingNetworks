#!/bin/sh

#SBATCH --partition=general
#SBATCH --qos=long
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:turing:1
#SBATCH --mem-per-cpu=4000
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

cd NeuralResponseRankingDAL/matchzoo


NUM_ITERS=100
BATCH_SIZE_NON_UDC=50

#========================================#
#               DMN                      #
#========================================#

TRAINING_TYPE='None'

#--------------------------------#
#       Train on MSDialog        #
#--------------------------------#

# ms -> udc
srun python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --train_batch_size $BATCH_SIZE_NON_UDC
srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters $NUM_ITERS --predict_ood False
# ms -> apple
srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --train_batch_size $BATCH_SIZE_NON_UDC
srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters $NUM_ITERS --predict_ood False


#--------------------------------#
#       Train on UDC             #
#--------------------------------#

# udc -> ms
srun python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False
srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters $NUM_ITERS --predict_ood False
# udc -> apple
srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters $NUM_ITERS --predict_ood False


#--------------------------------#
#       Train on Apple           #
#--------------------------------#

# apple -> ms
srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --train_batch_size $BATCH_SIZE_NON_UDC
srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters $NUM_ITERS --predict_ood False
# apple -> udc
srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --train_batch_size $BATCH_SIZE_NON_UDC
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters $NUM_ITERS --predict_ood False

#========================================#
#               DMN-ADL                  #
#========================================#

TRAINING_TYPE='DMN-ADL'

#--------------------------------#
#       Train on MSDialog        #
#--------------------------------#

# ms -> udc
srun python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --train_batch_size $BATCH_SIZE_NON_UDC
srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 1$NUM_ITERS --predict_ood False
# ms -> apple
srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --train_batch_size $BATCH_SIZE_NON_UDC
srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 1$NUM_ITERS --predict_ood False


#--------------------------------#
#       Train on UDC             #
#--------------------------------#

# udc -> ms
srun python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False
srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 1$NUM_ITERS --predict_ood False
# udc -> apple
srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters 1$NUM_ITERS --predict_ood False


#--------------------------------#
#       Train on Apple           #
#--------------------------------#

# apple -> ms
srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --train_batch_size $BATCH_SIZE_NON_UDC
srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 1$NUM_ITERS --predict_ood False
# apple -> udc
srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --train_batch_size $BATCH_SIZE_NON_UDC
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters 1$NUM_ITERS --predict_ood False

#========================================#
#               DMN-MTL                  #
#========================================#

TRAINING_TYPE='DMN-MTL'

#--------------------------------#
#       Train on MSDialog        #
#--------------------------------#

# ms -> udc
srun python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --train_batch_size $BATCH_SIZE_NON_UDC
srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 2$NUM_ITERS --predict_ood False
# ms -> apple
srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --train_batch_size $BATCH_SIZE_NON_UDC
srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 2$NUM_ITERS --predict_ood False


#--------------------------------#
#       Train on UDC             #
#--------------------------------#

# udc -> ms
srun python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False
srun python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 2$NUM_ITERS --predict_ood False
# udc -> apple
srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train 0 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters 2$NUM_ITERS --predict_ood False


#--------------------------------#
#       Train on Apple           #
#--------------------------------#

# apple -> ms
srun python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --train_batch_size $BATCH_SIZE_NON_UDC
srun python main_conversation_qa.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 2$NUM_ITERS --predict_ood False
# apple -> udc
srun python main_conversation_qa.py --phase train --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --domain_to_train 1 --domain_training_type $TRAINING_TYPE --num_iters $NUM_ITERS --predict False --train_batch_size $BATCH_SIZE_NON_UDC
srun python main_conversation_qa.py --phase predict --model_file config/udc_apple/dmn_cnn_udc_apple.config --or_cmd True --test_weights_iters 2$NUM_ITERS --predict_ood False

