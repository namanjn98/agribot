import pandas as pd
import string
import word2vec_m as wv
import weather_api
import sys
import data_cleaner

dimen = int(sys.argv[1])
k = int(sys.argv[2])
a = float(sys.argv[3])
pca_text = str(sys.argv[4])

#Chatbot
print '\nHi!, I am AgriBot'
print 'I am here to help you out with agriculture queries.\n'
print 'May I know your name?'

name = raw_input('')
name = str(name)
name = name.split(' ')[0]

print '\nSo %s, How can I help you?'%(name)
query = raw_input('')

print '\nSure, which district are you from?'
district = raw_input('')

print '\nAnd which state?'
state = raw_input('')

print '\nGive me a second\n\n'


#### Answer #####
query = query.lower()
input_list = [district,state,query]
query = data_cleaner.sentence_cleaner(query)


# weather_api.daily(district+','+state)
if 'weather' in query:
    weat = weather_api.daily(district+','+state)
    print '%s\n'%(weat)
    print 'Would you like to the weather forecast for the coming week ?\n'
    week = raw_input('').lower()
    if week == 'yes':
        weat = weather_api.weekly(district+','+state)
        print weat
    
    print '\nThank You for chatting'
    print '\nFor further information, contact KCC'

else:
    u, v, t, new_maharashtra, maharashtra = wv.pre('all_files.csv')
    ind = wv.test_query(u, v, t, input_list,dimen,k,a,pca=pca_text)

    query_list = query.split()
    fin_index = wv.entity(ind,query_list,maharashtra)

    if fin_index == -1:
        print "I apologise for the inconvenience but I may not be able to help."
    else:
        wv.print_ans(ind, maharashtra, k)

    print '\nThank You for chatting'
    print '\nFor further information, contact KCC'





