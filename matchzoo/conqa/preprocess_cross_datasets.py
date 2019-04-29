# /bin/python2.7
import sys
import pandas as pd
from IPython import embed

if __name__ == '__main__':
	if len(sys.argv) < 3:
		print 'please input params: data1 (source) and data2 (target) (udc or ms_v2)'
		exit(1)
	data_name_1 = sys.argv[1] # ms or udc or ms_v2 (source)
	data_name_2 = sys.argv[2] # ms or udc or ms_v2 (target)	

	basedir_1 = '../../data/' + data_name_1 + '/ModelInput/'
	cur_data_dir_1 = basedir_1 + 'dmn_model_input/'
	word_dict_1 = pd.read_csv(
		cur_data_dir_1+"word_dict.txt", sep=" ", names=["word", "id"]).\
		set_index("word").to_dict()['id']

	basedir_2 = '../../data/' + data_name_2 + '/ModelInput/'
	cur_data_dir_2 = basedir_2 + 'dmn_model_input/'
	word_dict_2 = pd.read_csv(
		cur_data_dir_2+"word_dict.txt", sep=" ", names=["word", "id"]).\
		set_index("id").to_dict()['word']
	
	new_file = []
	with open(cur_data_dir_2+'corpus_preprocessed.txt') as f:
		for line in f:
			splited = line.split("\t")			
			indexes_in_src = []
			for word_index in splited[2].split(" "):
				if(word_index != ''):
					word = word_dict_2[int(word_index)]
					if(word in word_dict_1):
						index_in_source_domain = word_dict_1[word]
						indexes_in_src.append(str(index_in_source_domain))
			if(len(indexes_in_src)>0):
				new_line=splited[0] + "\t" + splited[1]+ "\t" + ' '.join(indexes_in_src) + "\t\n"
			new_file.append(new_line)

	with open(cur_data_dir_1+'corpus_preprocessed_ood.txt','w') as f:
		for line in new_file:
			f.write(line)
