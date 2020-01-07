#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=8192
#SBATCH --mail-type=END

module use /opt/insy/modulefiles
module load cuda/10.0 cudnn/10.0-7.4.2.24

cd NeuralResponseRankingDAL/matchzoo


srun --partition=general --qos=short --time=00:15:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=4096 --mail-type=END python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple_no_train.config --or_cmd True --predict False

srun --partition=general --qos=short --time=0:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=8096 --mail-type=END python rep_calculations.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple_no_train.config --or_cmd True --predict_ood False --test_weight_iters 1

srun --partition=general --qos=short --time=2:00:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:turing:1 --mem-per-cpu=4096 --mail-type=END python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train 0 --num_iters 100 --predict False --train_batch_size 50

srun --partition=general --qos=short --time=2:00:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:turing:1 --mem-per-cpu=4096 --mail-type=END python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple_only_clf.config --or_cmd True --domain_training_type 'DMN-ADL' --num_iters 100 --test_weights_iters 100 --predict False

srun --partition=general --qos=short --time=0:10:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=4096 --mail-type=END python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --domain_to_train -1 --num_iters $NUM_ITERS --predict True

srun --partition=general --qos=short --time=0:10:00 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=4096 --mail-type=END python preprocess_cross_datasets.py ms_udc apple
srun --partition=general --qos=short --time=0:10:00 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=4096 --mail-type=END python preprocess_cross_datasets.py ms_apple udc
srun --partition=general --qos=short --time=0:10:00 --ntasks=1 --cpus-per-task=1 --mem-per-cpu=4096 --mail-type=END python preprocess_cross_datasets.py udc_apple apple

# srun --partition=general --qos=short --time=0:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=4096 --mail-type=END python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 1$NUM_ITERS --predict_ood False

srun --partition=general --qos=short --time=0:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=8096 --mail-type=END python rep_calculations.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 10011111 --predict_ood False

srun --partition=general --qos=short --time=0:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=8096 --mail-type=END python rep_calculations.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 10011111 --predict_ood False

srun --partition=general --qos=short --time=0:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=8096 --mail-type=END python rep_calculations.py --phase predict --model_file config/ms_apple/dmn_cnn_ms_apple.config --or_cmd True --test_weights_iters 2200 --predict_ood False

scp -r -o ProxyJump=gustavopenha@linux-bastion.tudelft.nl insy-login:/tudelft.net/staff-umbrella/domaincss/NeuralResponseRankingDAL/data/ms_udc/ModelRes/dmn_cnn.predict_ms.test.txt .

scp -r -o ProxyJump=gustavopenha@linux-bastion.tudelft.nl insy-login:/tudelft.net/staff-umbrella/domaincss/NeuralResponseRankingDAL/data/ms_udc/ModelRes/dmn_cnn.predict_udc.test.txt .

scp -r -o ProxyJump=gustavopenha@linux-bastion.tudelft.nl insy-login:/tudelft.net/staff-umbrella/domaincss/NeuralResponseRankingDAL/data/ms_udc/ModelRes/q_rep.pickle .

srun --partition=general --qos=short --time=2:00:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:turing:1 --mem-per-cpu=4096 --mail-type=END python main_conversation_qa.py --phase train --model_file config/ms_apple/dmn_cnn_ms_apple_only_clf.config --or_cmd True --domain_training_type 'DMN-ADL' --num_iters 100 --test_weights_iters 110011111 --predict False


#MSDialog

##	TRAIN

srun --partition=general --qos=short --time=2:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=4096 --mail-type=END python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_to_train 0 --num_iters 100 --predict False

srun --partition=general --qos=short --time=2:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=4096 --mail-type=END python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_training_type 'DMN-ADL' --domain_to_train 0 --num_iters 100 --predict False

srun --partition=general --qos=short --time=2:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=4096 --mail-type=END python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_training_type 'DMN-MTL' --domain_to_train 0 --num_iters 100 --predict False

##	EVAL

srun --partition=general --qos=short --time=0:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=2048 --mail-type=END python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 100

srun --partition=general --qos=short --time=0:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=2048 --mail-type=END python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 2100

srun --partition=general --qos=short --time=0:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=2048 --mail-type=END python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 1100

#UDC

##	TRAIN

srun --partition=general --qos=short --time=2:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=4096 --mail-type=END python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_to_train 1 --num_iters 100 --predict False

srun --partition=general --qos=short --time=2:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=4096 --mail-type=END python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_training_type 'DMN-ADL' --domain_to_train 1 --num_iters 100 --predict False

srun --partition=general --qos=short --time=2:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=4096 --mail-type=END python main_conversation_qa.py --phase train --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --domain_training_type 'DMN-MTL' --domain_to_train 1 --num_iters 100 --predict False

##	EVAL

srun --partition=general --qos=short --time=0:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=2048 --mail-type=END python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 100

srun --partition=general --qos=short --time=0:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=2048 --mail-type=END python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 2100

srun --partition=general --qos=short --time=0:30:00 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --mem-per-cpu=2048 --mail-type=END python main_conversation_qa.py --phase predict --model_file config/ms_udc/dmn_cnn_ms_udc.config --or_cmd True --test_weights_iters 1100

scp -r -o ProxyJump=gustavopenha@linux-bastion.tudelft.nl insy-login:/tudelft.net/staff-umbrella/domaincss/NeuralResponseRankingDAL/data/ms_udc/ModelRes/dmn_cnn.predict_udc.test.txtpredict_out .