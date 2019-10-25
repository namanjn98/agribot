import pandas as pd
import string
import word2vec_m as wv
from gensim.models import Word2Vec
import numpy as np
import sys
import data_cleaner
import similarity

dimen = int(sys.argv[1])
k = int(sys.argv[2])
a = float(sys.argv[3])
pca_text = str(sys.argv[4])

file_name = 'metric_test_dimen{}_k{}_a{}.csv'.format(dimen, k, a)
u, v, t, new_maharashtra, maharashtra = wv.pre('asf-all-train.csv')

train_data = new_maharashtra
test_data = pd.read_csv('asf-all-test.csv')['QueryText'] 

wv.word2vec_QAmodel(u, v, t, train_data, dimen, a, pca=pca_text)

district = 'Pune'
state = 'Maharashtra'
 
#### Answer #####

word2vec_value = np.load('word2vec_value.npy')
model = Word2Vec.load('model_word2vec.bin')

count_lesk = 0
count_jaccard = 0
count_lesk_threshold = 0
count_jaccard_threshold = 0
threshold_lesk = 0.8
threshold_jaccard = 0.7
total_count = 0

vayu = []
c = 0


for i,query in enumerate(list(test_data)):
    c+=1
    print c

    query = query.lower()
    input_list = [district,state,query]

    ind = wv.test_metric(u,v,t,query,dimen,k,a,model,word2vec_value,pca=pca_text)
    pdf = maharashtra.reset_index()

    query_list = data_cleaner.sentence_cleaner(query)
    fin_index = wv.entity(ind, query_list, pdf)

    lesk_score = similarity.compute_lesk_score(query, pdf['Query'][ind[fin_index]])
    jaccard_score = similarity.compute_jaccard_sim(query, pdf['Query'][ind[fin_index]])

    if lesk_score > threshold_lesk:
          count_lesk_threshold += 1

    if lesk_score > 0:
          count_lesk += 1

    if jaccard_score > threshold_jaccard:
          count_jaccard_threshold += 1

    if jaccard_score > 0:
          count_jaccard += 1

print "total count:", total_count
print "count_jaccard_threshold: ", count_jaccard_threshold
print "count_lesk_threshold: ", count_lesk_threshold

