import pandas as pd
import string
import word2vec_m as wv
import sys


dimen = int(sys.argv[1])
a = float(sys.argv[2])
pca_text = str(sys.argv[3])

u, v, t, new_india, india = wv.pre('all_files.csv')#('all_files.csv')
wv.word2vec_QAmodel(u,v,t,new_india,dimen,a,pca=pca_text)

