# /bin/python2.7
import sys
import pandas as pd
from IPython import embed

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print 'please input params: data1 (source) and data2 (target) embeddings'
		exit(1)
	data_name_1 = sys.argv[1] # ms or udc or ms_v2 (source)
	data_name_2 = sys.argv[2] # ms or udc or ms_v2 (target)	

	basedir_1 = '../../data/' + data_name_1 + '/ModelInput/'
	cur_data_dir_1 = basedir_1 + 'dmn_model_input/'
	word_dict_1 = pd.read_csv(
		cur_data_dir_1+"word_dict.txt", sep=" ", names=["word", "id"]).\
		set_index("id").to_dict()['word']

	basedir_2 = '../../data/' + data_name_2 + '/ModelInput/'
	cur_data_dir_2 = basedir_2 + 'dmn_model_input/'
	word_dict_2 = pd.read_csv(
		cur_data_dir_2+"word_dict.txt", sep=" ", names=["word", "id"]).\
		set_index("word").to_dict()['id']
	word_dict_2_reverse = pd.read_csv(
		cur_data_dir_2+"word_dict.txt", sep=" ", names=["word", "id"]).\
		set_index("id").to_dict()['word']

	embeddings = {}
	with open(basedir_1+'cut_embed_mikolov_200d_no_readvocab.txt') as f:
		for line in f:
			splited = line.split(" ")
			word_index = int(splited[0])
			word = word_dict_1[int(word_index)]
			if(word in word_dict_2):
				embeddings[word] = " " + " ".join(splited[1:])
	new_file = []
	with open(basedir_2+'cut_embed_mikolov_200d_no_readvocab.txt') as f:
		for line in f:
			splited = line.split(" ")
			word_index = int(splited[0])
			if(word_index not in word_dict_2_reverse):
				new_file.append(line)
				print("?")
			else:
				new_file.append(str(word_index) + embeddings[word_dict_2_reverse[word_index]])
	
	with open(basedir_2+'cut_embed_mikolov_200d_no_readvocab_from_'+data_name_1+'.txt','w') as f:
		for line in new_file:
			f.write(line)
