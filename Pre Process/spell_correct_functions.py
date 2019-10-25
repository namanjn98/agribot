import re
from collections import Counter
import pandas as pd
import sqlite3

DATABASE='edit_distance.db'
conn =  sqlite3.connect(DATABASE)
conn.execute('CREATE TABLE IF NOT EXISTS edit_table_zero(INPUT_WORD TEXT , EDITED_WORD TEXT, UNIQUE(INPUT_WORD, EDITED_WORD))')
conn.execute('CREATE TABLE IF NOT EXISTS edit_table_one(INPUT_WORD TEXT , EDITED_WORD TEXT, UNIQUE(INPUT_WORD, EDITED_WORD))')
conn.execute('CREATE TABLE IF NOT EXISTS edit_table_two(INPUT_WORD TEXT , EDITED_WORD TEXT, UNIQUE(INPUT_WORD, EDITED_WORD))')
cur = conn.cursor()

dict_number = {0:'zero', 1:'one', 2:'two'}

def make_list_word(in_word, edit_distance_length):
    all_rows = cur.execute("SELECT * FROM edit_table_{}".format(dict_number[edit_distance_length])+ " WHERE INPUT_WORD = (?)",(in_word,))
    list_edit_word = []
    for row in all_rows:
        list_edit_word.append(row[1])
    if len(list_edit_word)== 0:
        flag = 0
    else:
        flag = 1
    return list_edit_word, flag


def add_words_db(in_word, list_edit_word, edit_distance_length):
    for word in list_edit_word:
        conn.execute("INSERT OR IGNORE INTO edit_table_{}".format(dict_number[edit_distance_length])+"(INPUT_WORD, EDITED_WORD) VALUES(?,?)",(in_word, word))
        conn.commit()


#  Create a list of agricultural words
crop_names = pd.read_csv('Cropnames_Indianlanguages.csv')
english_word_list = list(crop_names['English'])
crop_common_name = []
crop_common_name_dict = {}
for i,w in enumerate(list(crop_names['Hindi'])):
    w = str(w).lower()
    crop_common_name.append(w)
    eng_word = str(english_word_list[i]).lower()
    crop_common_name_dict[w] = eng_word


def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('corpus_final.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N


def correction(word): 
    if word in crop_common_name:
        return(crop_common_name_dict[word])
    else:
        "Most probable spelling correction for word."
        word_spell_c = max(candidates(word), key=P)   
        return word_spell_c


def candidates(word):
    edit = {0: [], 1: [], 2: []}
    flag = 0
    for i in edit:
        t, f = make_list_word(word, i)
        edit[i] = set(t)
        flag += f
    if flag == 0:
        edit = {0: [], 1: [], 2: []}
        edit[0] = known([word])
        edit[1] = known(edits1(word)) 
        edit[2] = known(edits2(word))
        for i in edit:
            add_words_db(word, list(edit[i]), i)
    "Generate possible spelling corrections for word."
    return (edit[0] or edit[1] or edit[2] or [word])


def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)


def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)


def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))