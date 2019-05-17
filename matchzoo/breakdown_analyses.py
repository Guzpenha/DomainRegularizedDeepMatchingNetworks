import pandas as pd
from IPython import embed
import sys
import numpy as np
import random
from scipy import spatial, stats
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import operator
import pickle
from pprint import pprint

def calculate_map(r):
    def map(y_true, y_pred, rel_threshold=0):
        s = 0.
        y_true = (np.squeeze(y_true).tolist())
        y_pred = (np.squeeze(y_pred).tolist())
        c = zip(y_true, y_pred)
        random.shuffle(c)
        c = sorted(c, key=lambda x:x[1], reverse=True)
        ipos = 0
        for j, (g, p) in enumerate(c):
            if g > rel_threshold:
                ipos += 1.
                s += ipos / ( j + 1.)
        if ipos == 0:
            s = 0.
        else:
            s /= ipos
        return s
    return map(r["label"].tolist(), r["score"].tolist())

def get_query_and_doc_words(path):
    word_dict_path = path+"word_dict.txt"
    corpus_path = path+"corpus_preprocessed.txt"

    word_to_id_dict = pd.read_csv(
        word_dict_path, sep=" ", names=["word", "id"], keep_default_na=False).\
        set_index("word").to_dict()['id']    
    id_to_word_dict = {v: k for k, v in word_to_id_dict.iteritems()}    
    corpus = {}
    query_turn={}
    with open(corpus_path) as f:
        for line in f:
            splited = line.split("\t")
            query_turn[splited[0]] = int(splited[1])
            for word_index in splited[2].split(" "):
                if(splited[0] not in corpus):
                    corpus[splited[0]] = []
                if(word_index!=''):
                    corpus[splited[0]].append(int(word_index))
    return word_to_id_dict, id_to_word_dict, corpus, query_turn

# Read Embedding File
def read_embedding(filename):
    embed = {}
    for line in open(filename):
        line = line.strip().split()
        embed[int(line[0])] = map(float, line[1:])
    print '[%s]\n\tEmbedding size: %d' % (filename, len(embed))
    return embed

# python breakdown_analyses.py dmn_cnn.predict_ms.test.txtpredict_in /Users/gustavopenha/phd/emnlp19/NeuralResponseRanking/data/ms_v2/ModelInput/dmn_model_input/ /Users/gustavopenha/phd/emnlp19/NeuralResponseRanking/data/udc/ModelInput/dmn_model_input/ /Users/gustavopenha/phd/emnlp19/NeuralResponseRanking/data/ms_udc/ModelInput/dmn_model_input/
# python breakdown_analyses.py /Users/gustavopenha/phd/emnlp19/NeuralResponseRanking/data/ms_udc/ModelRes/dmn_cnn.predict_ms_and_udc.test.txt /Users/gustavopenha/phd/emnlp19/NeuralResponseRanking/data/ms_v2/ModelInput/dmn_model_input/ /Users/gustavopenha/phd/emnlp19/NeuralResponseRanking/data/udc/ModelInput/dmn_model_input/ /Users/gustavopenha/phd/emnlp19/NeuralResponseRanking/data/ms_udc/ModelInput/dmn_model_input/
# python breakdown_analyses.py /Users/gustavopenha/phd/emnlp19/NeuralResponseRanking/data/ms_apple/ModelRes/dmn_cnn.predict_ms_and_apple.test.txt /Users/gustavopenha/phd/emnlp19/NeuralResponseRanking/data/ms_v2/ModelInput/dmn_model_input/ /Users/gustavopenha/phd/emnlp19/NeuralResponseRanking/data/apple/ModelInput/dmn_model_input/ /Users/gustavopenha/phd/emnlp19/NeuralResponseRanking/data/ms_apple/ModelInput/dmn_model_input/

if __name__ == '__main__':
    if len(sys.argv) < 7:
        print('please input params: <trec_eval_file> <d1_folder> <d2_folder> <preds_folder> <tsne_file_suffix> <rep_to_use>')
        exit(1)
    path = sys.argv[1] # trec eval file path
    path_d1 = sys.argv[2] # trec eval file path
    path_d2 = sys.argv[3] # trec eval file path
    path_pred = sys.argv[4] 
    tsne_name = sys.argv[5] 
    rep_to_use = sys.argv[6] 

    df = pd.read_csv(path, sep="\t", names=["Q","_", "D", "rank", "score", "model", "label"])
    df_map = df.groupby(["Q"])['label','score']\
        .apply(lambda r,f = calculate_map: f(r)).reset_index()
    df_map.columns = ["Q", "map"]
    print("MEAN: ", df_map["map"].mean())

    preds_w_to_id, preds_id_to_w, preds_corpus, query_turn =  get_query_and_doc_words(path_pred)
    # d1_w_to_id, d1_id_to_w, d1_corpus , _ =  get_query_and_doc_words(path_d1)
    # d2_w_to_id, d2_id_to_w, d2_corpus, _ =  get_query_and_doc_words(path_d2)
    df_turns = pd.DataFrame.from_dict(query_turn, orient='index').reset_index()
    df_turns.columns = ['Q', 'turn']
    df_map = df_map.merge(df_turns, on='Q')

    utt_lengths = {}
    for k,v in preds_corpus.iteritems():
        utt_lengths[k] = len(v)
    df_map['utterance_length'] = df_map.apply(lambda r,m=utt_lengths: m[r['Q']],axis=1)
    
    analyze_query_representations=True
    if(analyze_query_representations):
        with open(path_pred.split('ModelInput')[0]+'ModelRes/q_rep.pickle', 'rb') as handle:
            utterances_w_emb = pickle.load(handle)
        reps = []
        qids = []

        rep_used = rep_to_use
        # rep_used = 'turn_1'
        # rep_used = 'match_rep'

        for k in utterances_w_emb.keys():
            if(rep_used=='turn_1'):
                rep = []
                for turn_rep in utterances_w_emb[k].keys():
                    rep = rep + utterances_w_emb[k][turn_rep].flatten().tolist()
            else:
                rep = utterances_w_emb[k][rep_used].flatten().tolist()

            reps.append(rep[0:60])
            qids.append(k)
        print("each utterance has " + str(len(rep))+" dimensions")
        del(utterances_w_emb)
        print("computing PCA")
        pca_50 = PCA(n_components=50)
        rep_pca = pca_50.fit_transform(reps)
        print("finished PCA")
        tsne = TSNE(n_components=2, verbose=1, perplexity=30, n_iter=500, learning_rate=100)
        tsne_results = tsne.fit_transform(rep_pca)

        df_tsne = pd.DataFrame(qids)
        df_tsne.columns = ["Q"]
        df_tsne['x-tsne'] = tsne_results[:,0]
        df_tsne['y-tsne'] = tsne_results[:,1]
        df_tsne_map = df_tsne.merge(df_map, on=['Q'])
        def get_domain_from_query(r):
            num = int(r["Q"].split("Q")[1])

            if( num > 9900000):
                domain="UDC"
            elif(num > 46774):
                domain="Apple"
            else:
                domain="MSDialog"
            # ms->udc
            # if( num > 9900000):
            #     domain="Apple"
            # elif(num > 574593):
            #     domain="UDC"
            # else:
            #     domain="MSDialog"
            return domain        

        if ('ms_v2' in path):
            cat_df = pd.read_csv("../data/ms_v2/ModelInput/ms_v2_categories.csv")
            queries_to_cat = {}
            for idx, row in cat_df.iterrows():
                queries_to_cat[row['Q']]=row['category']            
            df_tsne_map["domain"] = df_tsne_map.apply(lambda r, f = queries_to_cat: f[r["Q"]], axis=1)
        else:
            df_tsne_map["domain"] = df_tsne_map.apply(lambda r, f = get_domain_from_query: f(r), axis=1)
        # embed()
        df_tsne_map["ap"] = df_tsne_map["map"]
        df_tsne_map['lg_u_length'] = np.log(df_tsne_map.utterance_length)
        df_tsne_map[df_tsne_map['utterance_length']!=0].to_csv("tnse"+tsne_name+".csv", index=False, header=True)
        exit()
        # embed()
        flatten_rep = np.matrix(flatten_rep)
        for i in range(0, flatten_rep.shape[1], flatten_rep.shape[1]/10):
            turn_rep = flatten_rep[:,i:i+flatten_rep.shape[1]/10]
            print("computing PCA")
            pca_50 = PCA(n_components=50)
            rep_pca = pca_50.fit_transform(turn_rep)
            print("finished PCA")
            tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
            tsne_results = tsne.fit_transform(rep_pca)

            df_tsne_map['x-tsne-turn-'+str(i)] = tsne_results[:,0]
            df_tsne_map['y-tsne-turn-'+str(i)] = tsne_results[:,1]
        
        df_tsne_map[df_tsne_map['utterance_length']!=0].to_csv("tmp/tnse.csv", index=False, header=True)

        # flatten_rep = np.matrix(flatten_rep)
        # for i in range(flatten_rep.shape[1]):
        #     df_tsne_map[str(i)] = flatten_rep[:,i]
        # pd.melt(df_tsne_map.sample(n=1000).sort_values("domain")\
        #     .iloc[0:,4:], id_vars=['domain']).to_csv("tmp/flatten.csv")

        ncols = len(df_tsne_map.columns)
        for i in range(50):
            df_tsne_map['pca_'+str(i)] = rep_pca[:,i]

        centroid_PCA = df_tsne_map[df_tsne_map["domain"] == 'MSDialog'].iloc[:,ncols:].mean().values
        q_sim_to_centroid = []
        for idx,r in df_tsne_map.iterrows():
            sim = cosine_similarity([r.iloc[ncols:].values.tolist()], [centroid_PCA.tolist()])[0][0]
            q_sim_to_centroid.append([r["Q"], sim, r["domain"]])
        sim_df = pd.DataFrame(q_sim_to_centroid, columns = ["Q", "sim", "domain"])

        print(sim_df.sort_values("sim")[0:10])
        print(sim_df.sort_values("sim")[-10:])

        k_similar = 200
        queries_text = []
        word_count = {}
        for q in sim_df.sort_values("sim")[0:k_similar]["Q"]:
            query = ""
            for widx in preds_corpus[q]:
                word = preds_id_to_w[widx]
                query += word + " "
                if word not in word_count:
                    word_count[word]=0
                word_count[word]+=1
            queries_text.append(query)
        w_count_sorted = sorted(word_count.items(), key=operator.itemgetter(1))
        print("top "+str(k_similar)+" dissimilar")
        pprint(w_count_sorted[-50:])
        # print(queries_text[0:10])

        queries_text = []
        word_count = {}
        for q in sim_df.sort_values("sim")[-k_similar:]["Q"]:
            query = ""
            for widx in preds_corpus[q]:
                word = preds_id_to_w[widx]
                query += word + " "
                if word not in word_count:
                    word_count[word]=0
                word_count[word]+=1
            queries_text.append(query)
        w_count_sorted = sorted(word_count.items(), key=operator.itemgetter(1))
        print("top "+str(k_similar)+" similar")
        pprint(w_count_sorted[-50:])
        # print(queries_text[0:10])

        queries_text = []
        word_count = {}
        for q in sim_df[sim_df["domain"]=="UDC"].sort_values("sim")[-200:]["Q"]:
            query = ""
            for widx in preds_corpus[q]:
                word = preds_id_to_w[widx]
                query += word + " "
                if word not in word_count:
                    word_count[word]=0
                word_count[word]+=1
            queries_text.append(query)
        w_count_sorted = sorted(word_count.items(), key=operator.itemgetter(1))
        print("similar from UDC only")
        pprint(w_count_sorted[-50:])
        for q in queries_text[0:10]:
            pprint(q)
        print(queries_text[0:10])

    analyze_sim_to_domains=False
    if(analyze_sim_to_domains):
        print("Starting analyses of queries distances to representative words.")
        
        for distance in [spatial.distance.euclidean, spatial.distance.cosine, spatial.distance.correlation]:
            print(distance)

            embedding = read_embedding(path_pred.split("dmn_model_input/")[0]+'cut_embed_mikolov_200d_no_readvocab.txt')
            representative_0 = np.array(embedding[preds_w_to_id['ubuntu']])
            representative_1 = np.array(embedding[preds_w_to_id['microsoft']])

            percentage = 20
            best_performing_queries = df_map.sort_values("map")["Q"].values[-int(df_map.shape[0]/percentage):]
            worst_performing_queries = df_map.sort_values("map")["Q"].values[0:int(df_map.shape[0]/percentage)]

            d1_avg_sim_best = 0
            d2_avg_sim_best = 0
            d1_sims_best = []
            d2_sims_best = []
            count = 0
            for q in best_performing_queries:
                query = [preds_id_to_w[term] for term in preds_corpus[q]]
                embeddings = np.matrix([embedding[term] for term in preds_corpus[q]])
                if embeddings.shape[1] !=0:
                    count+=1
                    q_avg_embed = embeddings.mean(axis=0)

                    d1_avg_sim_best+= distance(q_avg_embed, representative_0)
                    d1_sims_best.append( distance(q_avg_embed, representative_0))
                    d2_avg_sim_best+= distance(q_avg_embed, representative_1)
                    d2_sims_best.append( distance(q_avg_embed, representative_1))

            d1_avg_sim_best = d1_avg_sim_best/count
            d2_avg_sim_best = d2_avg_sim_best/count

            d1_avg_sim_worst = 0
            d2_avg_sim_worst = 0
            d1_sims_worst = []
            d2_sims_worst = []
            count = 0
            for q in worst_performing_queries:
                query = [preds_id_to_w[term] for term in preds_corpus[q]]
                embeddings = np.matrix([embedding[term] for term in preds_corpus[q]])
                if embeddings.shape[1] !=0:
                    count+=1
                    q_avg_embed = embeddings.mean(axis=0)
                    d1_avg_sim_worst+= distance(q_avg_embed, representative_0)
                    d1_sims_worst.append(distance(q_avg_embed, representative_0))
                    d2_avg_sim_worst+= distance(q_avg_embed, representative_1)            
                    d2_sims_worst.append(distance(q_avg_embed, representative_1))

            d1_avg_sim_worst = d1_avg_sim_worst/count
            d2_avg_sim_worst = d2_avg_sim_worst/count

            # print("Measure to representative_0 (ubuntu) of top queries:" + str(d1_avg_sim_best))
            # print("Measure to representative_0 (ubuntu) of bottom queries:" + str(d1_avg_sim_worst))
            # print("\n")
            print("Measure to representative_1 (microsoft) of top queries:" + str(d2_avg_sim_best))
            print("Measure to representative_1 (microsoft) of bottom queries:" + str(d2_avg_sim_worst))

            t2, p2 = stats.ttest_ind(d2_sims_best, d2_sims_worst)
            print("t = " + str(t2))
            print("p = " + str(p2))

    analyze_oov=False
    if(analyze_oov):
        percentage = 10
        worst_performing_queries = df_map.sort_values("map")["Q"].values()[0:int(df_map.shape[0]/percentage)]
        best_performing_queries = df_map.sort_values("map")["Q"].values()[-int(df_map.shape[0]/percentage):]

        d1_count = 0
        d2_count = 0
        total = 0
        d1_percentages = []
        d2_percentages = []
        for query in worst_performing_queries:
            d1_by_query=0
            d2_by_query=0
            for term in preds_corpus[query]:
                word = preds_id_to_w[term]
                if word in d1_w_to_id:
                    d1_by_query+=1
                    d1_count+=1
                if word in d2_w_to_id:
                    d2_by_query+=1
                    d2_count+=1
                total+=1
            if(len(preds_corpus[query])>0):
                d1_percentages.append(float(d1_by_query)/len(preds_corpus[query]))
                d2_percentages.append(float(d2_by_query)/len(preds_corpus[query]))

        print("\n\n QUERIES ANALYSIS")
        print("Worst performing queries")
        print("percentage of query terms in d1 :"+str(float(d1_count)/total))
        print("percentage of query terms in d2 :"+str(float(d2_count)/total))
        print("avg percentages of terms d1: "+ str(float(sum(d1_percentages))/len(d1_percentages)))
        print("avg percentages of terms d2: "+ str(float(sum(d2_percentages))/len(d2_percentages)))

        d1_count = 0
        d2_count = 0
        total = 0
        d1_percentages = []
        d2_percentages = []
        for query in best_performing_queries:
            d1_by_query=0
            d2_by_query=0
            for term in preds_corpus[query]:
                word = preds_id_to_w[term]
                if word in d1_w_to_id:
                    d1_by_query+=1
                    d1_count+=1
                if word in d2_w_to_id:
                    d2_by_query+=1
                    d2_count+=1
                total+=1
            if(len(preds_corpus[query])>0):
                d1_percentages.append(float(d1_by_query)/len(preds_corpus[query]))
                d2_percentages.append(float(d2_by_query)/len(preds_corpus[query]))

        print("Best performing queries")
        print("percentage of query terms in d1 :"+str(float(d1_count)/total))
        print("percentage of query terms in d2 :"+str(float(d2_count)/total))
        print("avg percentages of terms d1: "+ str(float(sum(d1_percentages))/len(d1_percentages)))
        print("avg percentages of terms d2: "+ str(float(sum(d2_percentages))/len(d2_percentages)))

        worst_performing_documents = df[(df["Q"].isin(worst_performing_queries)) & (df["label"] == 1)]["D"].tolist()
        best_performing_documents = df[(df["Q"].isin(best_performing_queries)) & (df["label"] == 1)]["D"].tolist()

        d1_count = 0
        d2_count = 0
        total = 0
        d1_percentages = []
        d2_percentages = []
        for doc in worst_performing_documents:
            d1_by_query=0
            d2_by_query=0
            for term in preds_corpus[doc]:
                word = preds_id_to_w[term]
                if word in d1_w_to_id:
                    d1_by_query+=1
                    d1_count+=1    
                if word in d2_w_to_id:
                    d2_by_query+=1
                    d2_count+=1
                total+=1
            if(len(preds_corpus[doc])>0):
                d1_percentages.append(float(d1_by_query)/len(preds_corpus[doc]))
                d2_percentages.append(float(d2_by_query)/len(preds_corpus[doc]))
     
        print("\n\n RELEVANT DOCUMENTS ANALYSIS")
        print("Worst performing queries")
        print("percentage of correct doc terms in d1 :"+str(float(d1_count)/total))
        print("percentage of correct doc terms in d2 :"+str(float(d2_count)/total))
        print("avg percentages of terms d1: "+ str(float(sum(d1_percentages))/len(d1_percentages)))
        print("avg percentages of terms d2: "+ str(float(sum(d2_percentages))/len(d2_percentages)))
        d1_count = 0
        d2_count = 0
        total = 0
        d1_percentages = []
        d2_percentages = []
        for doc in best_performing_documents:
            d1_by_query=0
            d2_by_query=0
            for term in preds_corpus[doc]:
                word = preds_id_to_w[term]
                if word in d1_w_to_id:
                    d1_by_query+=1
                    d1_count+=1
                if word in d2_w_to_id:
                    d2_by_query+=1
                    d2_count+=1
                total+=1
            if(len(preds_corpus[doc])>0):
                d1_percentages.append(float(d1_by_query)/len(preds_corpus[doc]))
                d2_percentages.append(float(d2_by_query)/len(preds_corpus[doc]))

        print("Best performing queries")
        print("percentage of correct doc terms in d1 :"+str(float(d1_count)/total))
        print("percentage of correct doc terms in d2 :"+str(float(d2_count)/total))
        print("avg percentages of terms d1: "+ str(float(sum(d1_percentages))/len(d1_percentages)))
        print("avg percentages of terms d2: "+ str(float(sum(d2_percentages))/len(d2_percentages)))


        worst_performing_documents = df[(df["Q"].isin(worst_performing_queries))]["D"].tolist()
        best_performing_documents = df[(df["Q"].isin(best_performing_queries))]["D"].tolist()

        d1_count = 0
        d2_count = 0
        total = 0
        d1_percentages = []
        d2_percentages = []
        for doc in worst_performing_documents:
            d1_by_query=0
            d2_by_query=0
            for term in preds_corpus[doc]:
                word = preds_id_to_w[term]
                if word in d1_w_to_id:
                    d1_by_query+=1
                    d1_count+=1    
                if word in d2_w_to_id:
                    d2_by_query+=1
                    d2_count+=1
                total+=1
            if(len(preds_corpus[doc])>0):
                d1_percentages.append(float(d1_by_query)/len(preds_corpus[doc]))
                d2_percentages.append(float(d2_by_query)/len(preds_corpus[doc]))
     
        print("\n\n ALL DOCUMENTS ANALYSIS")
        print("Worst performing queries")
        print("percentage of correct doc terms in d1 :"+str(float(d1_count)/total))
        print("percentage of correct doc terms in d2 :"+str(float(d2_count)/total))
        print("avg percentages of terms d1: "+ str(float(sum(d1_percentages))/len(d1_percentages)))
        print("avg percentages of terms d2: "+ str(float(sum(d2_percentages))/len(d2_percentages)))
        d1_count = 0
        d2_count = 0
        total = 0
        d1_percentages = []
        d2_percentages = []
        for doc in best_performing_documents:
            d1_by_query=0
            d2_by_query=0
            for term in preds_corpus[doc]:
                word = preds_id_to_w[term]
                if word in d1_w_to_id:
                    d1_by_query+=1
                    d1_count+=1
                if word in d2_w_to_id:
                    d2_by_query+=1
                    d2_count+=1
                total+=1
            if(len(preds_corpus[doc])>0):
                d1_percentages.append(float(d1_by_query)/len(preds_corpus[doc]))
                d2_percentages.append(float(d2_by_query)/len(preds_corpus[doc]))

        print("Best performing queries")
        print("percentage of correct doc terms in d1 :"+str(float(d1_count)/total))
        print("percentage of correct doc terms in d2 :"+str(float(d2_count)/total))
        print("avg percentages of terms d1: "+ str(float(sum(d1_percentages))/len(d1_percentages)))
        print("avg percentages of terms d2: "+ str(float(sum(d2_percentages))/len(d2_percentages)))



