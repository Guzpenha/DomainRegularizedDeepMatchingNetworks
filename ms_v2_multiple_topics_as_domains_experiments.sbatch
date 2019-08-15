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
	for CATEGORY in 'Windows_Insider_Register,Outlook_Email,Office_Account,Movie_Maker,Skype_Windows_Desktop,Surface_Pro_4,Onedrive,Bing_Search,MSN,Outlook' 'Windows_Insider_Register,Outlook_Email,Office_Account,Movie_Maker,Skype_Windows_Desktop,Surface_Pro_4,Onedrive,Bing_Search,MSN,Outlook,IE_8,Lumia,NOT_FOUND,Microsoft_Band,Outlook_Contacts,Asha,Outlook_Preview,Outlook_Calendar,Bing,Surface_Pro_3' 'Windows_Insider_Register,Outlook_Email,Office_Account,Movie_Maker,Skype_Windows_Desktop,Surface_Pro_4,Onedrive,Bing_Search,MSN,Outlook,IE_8,Lumia,NOT_FOUND,Microsoft_Band,Outlook_Contacts,Asha,Outlook_Preview,Outlook_Calendar,Bing,Surface_Pro_3,Outlook_Messaging,Skype_iOS,Live_Mail,Skype_Windows_10,IE_7,Movies&TV,Defender,Windows_7,Word,Windows_Insider_Apps' 'Windows_Insider_Register,Outlook_Email,Office_Account,Movie_Maker,Skype_Windows_Desktop,Surface_Pro_4,Onedrive,Bing_Search,MSN,Outlook,IE_8,Lumia,NOT_FOUND,Microsoft_Band,Outlook_Contacts,Asha,Outlook_Preview,Outlook_Calendar,Bing,Surface_Pro_3,Outlook_Messaging,Skype_iOS,Live_Mail,Skype_Windows_10,IE_7,Movies&TV,Defender,Windows_7,Word,Windows_Insider_Apps,Office_Install,Skype_Mac,Windows_8.1,Access,Skype_Android,IE_10,Bing_Ads,Windows_Insider_Preview,Excel,Essentials_Gallery' 'Windows_Insider_Register,Outlook_Email,Office_Account,Movie_Maker,Skype_Windows_Desktop,Surface_Pro_4,Onedrive,Bing_Search,MSN,Outlook,IE_8,Lumia,NOT_FOUND,Microsoft_Band,Outlook_Contacts,Asha,Outlook_Preview,Outlook_Calendar,Bing,Surface_Pro_3,Outlook_Messaging,Skype_iOS,Live_Mail,Skype_Windows_10,IE_7,Movies&TV,Defender,Windows_7,Word,Windows_Insider_Apps,Office_Install,Skype_Mac,Windows_8.1,Access,Skype_Android,IE_10,Bing_Ads,Windows_Insider_Preview,Excel,Essentials_Gallery,Games_Windows_10,Media_Player,Bing_Apps,Onenote,IE_11,Onedrive_Business,Groove_Music,Windows_Mobile,Apps_Windows_10,Windows_Insider_Office' 'Windows_Insider_Register,Outlook_Email,Office_Account,Movie_Maker,Skype_Windows_Desktop,Surface_Pro_4,Onedrive,Bing_Search,MSN,Outlook,IE_8,Lumia,NOT_FOUND,Microsoft_Band,Outlook_Contacts,Asha,Outlook_Preview,Outlook_Calendar,Bing,Surface_Pro_3,Outlook_Messaging,Skype_iOS,Live_Mail,Skype_Windows_10,IE_7,Movies&TV,Defender,Windows_7,Word,Windows_Insider_Apps,Office_Install,Skype_Mac,Windows_8.1,Access,Skype_Android,IE_10,Bing_Ads,Windows_Insider_Preview,Excel,Essentials_Gallery,Games_Windows_10,Media_Player,Bing_Apps,Onenote,IE_11,Onedrive_Business,Groove_Music,Windows_Mobile,Apps_Windows_10,Windows_Insider_Office,Windows_RT_8.1,Windows_10,Office_Insider,Skype_Linux,Surface_accessories,MSE,PowerPoint,Windows_Insider_IE_Edge,Zune,Cortana_Preview' 'Windows_Insider_Register,Outlook_Email,Office_Account,Movie_Maker,Skype_Windows_Desktop,Surface_Pro_4,Onedrive,Bing_Search,MSN,Outlook,IE_8,Lumia,NOT_FOUND,Microsoft_Band,Outlook_Contacts,Asha,Outlook_Preview,Outlook_Calendar,Bing,Surface_Pro_3,Outlook_Messaging,Skype_iOS,Live_Mail,Skype_Windows_10,IE_7,Movies&TV,Defender,Windows_7,Word,Windows_Insider_Apps,Office_Install,Skype_Mac,Windows_8.1,Access,Skype_Android,IE_10,Bing_Ads,Windows_Insider_Preview,Excel,Essentials_Gallery,Games_Windows_10,Media_Player,Bing_Apps,Onenote,IE_11,Onedrive_Business,Groove_Music,Windows_Mobile,Apps_Windows_10,Windows_Insider_Office,Windows_RT_8.1,Windows_10,Office_Insider,Skype_Linux,Surface_accessories,MSE,PowerPoint,Windows_Insider_IE_Edge,Zune,Cortana_Preview,IE_9,Bing_Maps,Defender_Offline,Nokia,Skype_Web,MSN_Install,Surface_3,Surface_Laptop,Xbox_Music,Windows_Insider_Games' 'Windows_Insider_Register,Outlook_Email,Office_Account,Movie_Maker,Skype_Windows_Desktop,Surface_Pro_4,Onedrive,Bing_Search,MSN,Outlook,IE_8,Lumia,NOT_FOUND,Microsoft_Band,Outlook_Contacts,Asha,Outlook_Preview,Outlook_Calendar,Bing,Surface_Pro_3,Outlook_Messaging,Skype_iOS,Live_Mail,Skype_Windows_10,IE_7,Movies&TV,Defender,Windows_7,Word,Windows_Insider_Apps,Office_Install,Skype_Mac,Windows_8.1,Access,Skype_Android,IE_10,Bing_Ads,Windows_Insider_Preview,Excel,Essentials_Gallery,Games_Windows_10,Media_Player,Bing_Apps,Onenote,IE_11,Onedrive_Business,Groove_Music,Windows_Mobile,Apps_Windows_10,Windows_Insider_Office,Windows_RT_8.1,Windows_10,Office_Insider,Skype_Linux,Surface_accessories,MSE,PowerPoint,Windows_Insider_IE_Edge,Zune,Cortana_Preview,IE_9,Bing_Maps,Defender_Offline,Nokia,Skype_Web,MSN_Install,Surface_3,Surface_Laptop,Xbox_Music,Windows_Insider_Games,Xbox_Video,Bing_Safety,Surface_Studio,Edge,Skype_Xbox'
	do
		#========================================#
		#               DMN                      #
		#========================================#

		TRAINING_TYPE='None'
		srun python main_conversation_qa.py --phase train --model_file config/ms_v2/dmn_cnn_N_topics_as_domains.config --or_cmd True --test_categories $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0 --keras_random_seed $RANDOM_SEED
		srun python main_conversation_qa.py --phase predict --model_file config/ms_v2/dmn_cnn_N_topics_as_domains.config --or_cmd True --test_categories $CATEGORY --test_weights_iters ${NUM_ITERS} --predict_ood False

		for INPUT_TO_CLF in 'query_doc'
		do
			#========================================#
			#               DMN-ADL                  #
			#========================================#

			TRAINING_TYPE='DMN-ADL'
			srun python main_conversation_qa.py --phase train --model_file config/ms_v2/dmn_cnn_N_topics_as_domains.config --or_cmd True --test_categories $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0 --reset_clf_weights_iters $RESET_CLF_WEIGHTS --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED
			srun python main_conversation_qa.py --phase predict --model_file config/ms_v2/dmn_cnn_N_topics_as_domains.config --or_cmd True --test_categories $CATEGORY --test_weights_iters 1${NUM_ITERS}11111 --predict_ood False

			#========================================#
			#               DMN-MTL                  #
			#========================================#

			TRAINING_TYPE='DMN-MTL'
			srun python main_conversation_qa.py --phase train --model_file config/ms_v2/dmn_cnn_N_topics_as_domains.config --or_cmd True --test_categories $CATEGORY --domain_training_type $TRAINING_TYPE --num_iters ${NUM_ITERS} --predict False --domain_to_train 0 --input_to_domain_clf $INPUT_TO_CLF --keras_random_seed $RANDOM_SEED
			srun python main_conversation_qa.py --phase predict --model_file config/ms_v2/dmn_cnn_N_topics_as_domains.config --or_cmd True --test_categories $CATEGORY --test_weights_iters 2${NUM_ITERS}11111 --predict_ood False
		done
	done
done
