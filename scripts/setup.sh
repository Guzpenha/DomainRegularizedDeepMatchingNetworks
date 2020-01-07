git clone https://github.com/Guzpenha/NeuralResponseRankingDAL
pip install --user keras==2.1.6
# pip install --user tensorflow
pip install --user tensorflow-gpu
pip install --user tqdm
pip install --user IPython
pip install --user pandas==0.24.2
pip install --user sklearn

## apple files
mkdir NeuralResponseRankingDAL/data/apple/
mkdir NeuralResponseRankingDAL/data/apple/ModelInput
mkdir NeuralResponseRankingDAL/data/apple/ModelInput/dmn_model_input
mkdir NeuralResponseRankingDAL/data/apple/ModelRes
mkdir NeuralResponseRankingDAL/data/apple/ModelInput/dmn_prf_model_input_body
mkdir NeuralResponseRankingDAL/data/apple/ModelInput/word2vec_mikolov
cd NeuralResponseRankingDAL/data/apple/ModelInput/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-IThIeiJosXBc8Ep8ktF0ZnxzfBLEPzv' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-IThIeiJosXBc8Ep8ktF0ZnxzfBLEPzv" -O conv_search.zip && rm -rf /tmp/cookies.txt
unzip conv_search.zip
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1kp9xMbydOXtRoPul8IowT2gF7BjM_UMq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1kp9xMbydOXtRoPul8IowT2gF7BjM_UMq" -O cut_embed_mikolov_200d_no_readvocab.txt && rm -rf /tmp/cookies.txt
cd ../../../../
cd NeuralResponseRankingDAL/data/apple/ModelInput/dmn_model_input
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TPo5crFvHr2qF6Sq7U-L1fQ1QhjozH6H' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TPo5crFvHr2qF6Sq7U-L1fQ1QhjozH6H" -O dmn_model_input.zip && rm -rf /tmp/cookies.txt 
unzip dmn_model_input.zip
cd ../../../../

## ms_v2 files
cd NeuralResponseRankingDAL/data/ms_v2/ModelInput/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1R_c8b7Yi0wChA_du3eKDtnOGuYTqVhnY' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1R_c8b7Yi0wChA_du3eKDtnOGuYTqVhnY" -O MSDialog.tar.gz && rm -rf /tmp/cookies.txt
tar -xzf MSDialog.tar.gz
cd ../../../../
mv NeuralResponseRankingDAL/data/ms_v2/ModelInput/MSDialog/train.tsv NeuralResponseRankingDAL/data/ms_v2/ModelInput/train.txt
mv NeuralResponseRankingDAL/data/ms_v2/ModelInput/MSDialog/test.tsv NeuralResponseRankingDAL/data/ms_v2/ModelInput/test.txt
mv NeuralResponseRankingDAL/data/ms_v2/ModelInput/MSDialog/valid.tsv NeuralResponseRankingDAL/data/ms_v2/ModelInput/valid.txt
mkdir NeuralResponseRankingDAL/data/ms_v2/ModelInput/dmn_model_input
mkdir NeuralResponseRankingDAL/data/ms_v2/ModelRes
mkdir NeuralResponseRankingDAL/data/ms_v2/ModelInput/dmn_prf_model_input_body
mkdir NeuralResponseRankingDAL/data/udc/ModelInput/word2vec_mikolov
cd NeuralResponseRankingDAL/data/ms_v2/ModelInput/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1aWe63Lu2TtQphnnMorGrQWxqW2vfTkUn' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1aWe63Lu2TtQphnnMorGrQWxqW2vfTkUn" -O cut_embed_mikolov_200d_no_readvocab.txt && rm -rf /tmp/cookies.txt
cd ../../../../
cd NeuralResponseRankingDAL/data/ms_v2/ModelInput/dmn_model_input
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BAzoSeFhHCFt8JLk_Bwb65EJ1SXLxwco' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BAzoSeFhHCFt8JLk_Bwb65EJ1SXLxwco" -O dmn_model_input.zip && rm -rf /tmp/cookies.txt
unzip dmn_model_input.zip
cd ../../../../


## udc files
cd NeuralResponseRankingDAL/data/udc/ModelInput/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1UVocWK5rZkIPuPv8cUxaUtR16Bmg616V' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1UVocWK5rZkIPuPv8cUxaUtR16Bmg616V" -O ubuntu_data.zip && rm -rf /tmp/cookies.txt
unzip ubuntu_data.zip
cd ../../../../
mv NeuralResponseRankingDAL/data/udc/ModelInput/ubuntu_data/train.txt NeuralResponseRankingDAL/data/udc/ModelInput/train.txt
mv NeuralResponseRankingDAL/data/udc/ModelInput/ubuntu_data/test.txt NeuralResponseRankingDAL/data/udc/ModelInput/test.txt
mv NeuralResponseRankingDAL/data/udc/ModelInput/ubuntu_data/valid.txt NeuralResponseRankingDAL/data/udc/ModelInput/valid.txt
mkdir NeuralResponseRankingDAL/data/udc/ModelInput/dmn_model_input
mkdir NeuralResponseRankingDAL/data/udc/ModelRes
mkdir NeuralResponseRankingDAL/data/udc/ModelInput/dmn_prf_model_input_body
mkdir NeuralResponseRankingDAL/data/udc/ModelInput/word2vec_mikolov
cd NeuralResponseRankingDAL/data/udc/ModelInput/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1shLeJyhqew_S7Rcgk4NETPIPvkH8PmMs' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1shLeJyhqew_S7Rcgk4NETPIPvkH8PmMs" -O cut_embed_mikolov_200d_no_readvocab.txt && rm -rf /tmp/cookies.txt
cd ../../../../
mv NeuralResponseRankingDAL/data/udc/ModelInput/cut_embed_mikolov_200d_no_readvocab.txt NeuralResponseRankingDAL/data/udc/ModelInput/dmn_model_input/cut_embed_mikolov_200d_no_readvocab.txt
cd NeuralResponseRankingDAL/data/udc/ModelInput/dmn_model_input
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1VdzXjaljuH3L9YOFsDsnr1KLUdextYDI' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1VdzXjaljuH3L9YOFsDsnr1KLUdextYDI" -O dmn_model_input.zip && rm -rf /tmp/cookies.txt
unzip dmn_model_input.zip
cd ../../../../

## ms_apple files
cd NeuralResponseRankingDAL/data/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1TLf_oSuI2WMqC264cV3gy3bzTFVz8YKO' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1TLf_oSuI2WMqC264cV3gy3bzTFVz8YKO" -O ms_apple.zip && rm -rf /tmp/cookies.txt
unzip ms_apple.zip
rm ms_apple.zip

## udc_apple files
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1RzNiQxSPlCOKl54WYEpT4t34H1VNhKJK' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1RzNiQxSPlCOKl54WYEpT4t34H1VNhKJK" -O udc_apple.zip && rm -rf /tmp/cookies.txt
unzip udc_apple.zip
rm udc_apple.zip


## ms_udc files
mkdir NeuralResponseRankingDAL/data/ms_udc/
mkdir NeuralResponseRankingDAL/data/ms_udc/ModelInput
mkdir NeuralResponseRankingDAL/data/ms_udc/ModelInput/dmn_model_input
mkdir NeuralResponseRankingDAL/data/ms_udc/ModelRes
mkdir NeuralResponseRankingDAL/data/ms_udc/ModelInput/dmn_prf_model_input_body
mkdir NeuralResponseRankingDAL/data/ms_udc/ModelInput/word2vec_mikolov
cd NeuralResponseRankingDAL/data/ms_udc/ModelInput/dmn_model_input
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1U2-y49yCHl9afdAkjksGKnFtzaBTVAHj' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1U2-y49yCHl9afdAkjksGKnFtzaBTVAHj" -O dmn_model_input.zip && rm -rf /tmp/cookies.txt
unzip dmn_model_input.zip
cd ../../../../
cd NeuralResponseRankingDAL/data/ms_udc/ModelInput/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1-A7bR2qjgfyTwuZB_VuEKCIGRDHnok1C' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1-A7bR2qjgfyTwuZB_VuEKCIGRDHnok1C" -O domain_splits_test && rm -rf /tmp/cookies.txt  
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1c1CMRNt3GaU1RsjOnmp6PzdVKR-0Zn5V' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1c1CMRNt3GaU1RsjOnmp6PzdVKR-0Zn5V" -O domain_splits_train && rm -rf /tmp/cookies.txt
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=177AKU32lEDybjHObXmq3V69HaF1WzRKX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=177AKU32lEDybjHObXmq3V69HaF1WzRKX" -O cut_embed_mikolov_200d_no_readvocab.txt && rm -rf /tmp/cookies.txt
cd ../../../../


## mantis files
cd NeuralResponseRankingDAL/data/
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=16q2vN243oe5YJ_-pJY3wMQ-zpySRy5bc' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=16q2vN243oe5YJ_-pJY3wMQ-zpySRy5bc" -O mantis.zip && rm -rf /tmp/cookies.txt
unzip mantis.zip
rm mantis.zip
