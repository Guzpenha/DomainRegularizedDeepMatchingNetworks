#gen_w2v_mikolov.py
#preprocess_dmn.py
#config/mantis_10/dmn_cnn.config

mkdir NeuralResponseRanking/data/mantis_10
mkdir NeuralResponseRanking/data/mantis_10/ModelInput/
mkdir NeuralResponseRanking/data/mantis_10/ModelRes/
mkdir NeuralResponseRanking/data/mantis_10/ModelInput/dmn_model_input
mkdir NeuralResponseRanking/data/mantis_10/ModelInput/word2vec_mikolov
cd NeuralResponseRanking/data/mantis_10/ModelInput/word2vec_mikolov
git clone https://github.com/dav/word2vec.git
cd word2vec/src/ ; make
cd ../
#Ubuntu:
# wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BDo6owWCATtpULkcm5cjYnR2vA0E2bwL' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BDo6owWCATtpULkcm5cjYnR2vA0E2bwL" -O mantis_10.7z && rm -rf /tmp/cookies.txt
#MAC (gsed):
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BDo6owWCATtpULkcm5cjYnR2vA0E2bwL' -O- | gsed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BDo6owWCATtpULkcm5cjYnR2vA0E2bwL" -O mantis_10.7z && rm -rf /tmp/cookies.txt
7za x mantis_10.7z

cd ../../../
cd matchzoo/conqa/
python preprocess_dmn.py mantis_10
python gen_w2v_mikolov.py mantis_10 0 dmn_model_input
python gen_w2v_filtered.py ../../data/mantis_10/ModelInput/dmn_model_input/train_word2vec_mikolov_200d_no_readvocab.txt ../../data/mantis_10/ModelInput/dmn_model_input/word_dict.txt ../../data/mantis_10/ModelInput/cut_embed_mikolov_200d_no_readvocab.txt

cd ../
python main_conversation_qa.py --phase train --model_file config/mantis_10/dmn_cnn.config --or_cmd True --num_iters 200
python main_conversation_qa.py --phase predict --model_file config/mantis_10/dmn_cnn.config --or_cmd True --test_weights_iters 200