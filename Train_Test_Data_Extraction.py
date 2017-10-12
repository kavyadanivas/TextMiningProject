#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Title : Aspect Based Sentiment Analysis (Par 1 - Test data generation)
# Authors : Kavya Danivas, Lilia Fkaier
# Dataset : Annotated Trip Advisor dataset http://nemis.isti.cnr.it/~marcheggiani/datasets/

import time
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize, PunktSentenceTokenizer, RegexpTokenizer
from nltk.corpus import stopwords, words
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import state_union
from matplotlib.cbook import unique
from nltk import pos_tag
from nltk.corpus import wordnet
# import gensim.models
# from gensim.models import Word2Vec
import os
import json

ps = PorterStemmer()                #to normalize into the root word
lemmatizer=WordNetLemmatizer()      #to give similar meaning of a word
tokenizer_reg = RegexpTokenizer(r'\w+') #to tokenize + removing punctuations
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# User input paramters
sim_val_threshold = 0.4 # Similarity threshold
path = 'LargeTripadvisor/'
senti_threshold = 3 # between 1 to 5, if val > senti_threshold then pos, else neg
writeFileName = './trainDataVeryLarge.json' # output trained data file

# Predefine aspect terms in the file and associated aspect terms considered in the project
predefined_aspects = {"Rooms" : "room", "Cleanliness": "cleanliness",
                      "Value":"value", "Service":"service",
                      "Location": "location", "Sleep Quality":"sleep",
                      "Business service (e.g., internet access)":"internet", "Business service":"wireless","Check in / front desk":"reception"}  #Business service (e.g., internet access)', 'Check in / front desk'

noun = ["NN", "NNS", "NNP", "NNPS"]             #defining POS tags for words #Currently not been used in the code
adjective = ["JJ", "JJR", "JJS"]                #defining POS tags for words

stops=stopwords.words("english")
stops.extend(['.', ',', "'s", "n't", '!', '(', ')', '-', ':', '!', '?', '...', '..', '+', ';', '<', '>'])
stop_words = set(stops)
a=set(['not','nor','no','aren','haven','isn','doesn', 'hasn', 'wasn', 'mustn', 'didnt', 'didn', 'shouldn', 'mightn','weren'])
stop_words_modified=list(stop_words-a)
start_time = time.time()


# store aspect terms that are predefined
predefined_aspect_terms = predefined_aspects.keys()
predef_aspect_term_modified=unique(predefined_aspects.values())

# print aspect terms and categories that are predefined
print("Predefined Aspect Terms: ")
print(predefined_aspect_terms) #TODO: Clean the output

print("Defined Aspect Terms in the Project: ")
print(predef_aspect_term_modified) #TODO: Clean the output

start_time = time.time()
listing = os.listdir(path)

aspect_terms_found = []


'''
Function to write to a json file
Input : Data - Text and label
Output : void
'''
def writeToJsonFile(fileName, data_in):
    with open(fileName, 'w') as fp:
        json.dump(data_in, fp)


'''
Function for extracting aspect term
Input : review line (sentence)
Output : aspect word
'''
def extract_aspect_term(review_line):
    review_line = review_line.lower() # normalize the sentence to all lower case
    review_linewords = tokenizer_reg.tokenize(review_line)
    filtered_review_line = [wd for wd in review_linewords if not wd in stop_words_modified] # remove all stop words from the sentence
    # print(filtered_review_line)
    tagged = nltk.pos_tag(filtered_review_line)  # tag pos to words
    max_sim_val = 0
    asp_word_found = False
    adj_found = False
    for (w1, t1) in tagged: # loop through all the pos tagged words
        for wn in aspect_terms_found: #aspect terms found are the terms found in the review; not the pre defined
            if(wn=='Overall'):
                continue

            wnn=predefined_aspects[wn]
            sim_val = 0
            wn2 = lemmatizer.lemmatize(w1)
            try:
                wone = wordnet.synset(wnn + '.n.01')
                wtwo = wordnet.synset(wn2 + '.n.01')
                sim_val = wone.wup_similarity(wtwo)
            except Exception:
                continue

            if(sim_val > sim_val_threshold and sim_val > max_sim_val):
                 asp_word = wn
                 max_sim_val = sim_val
                 asp_word_found = True

        if(t1 in adjective):
            adj_found=True

    if(asp_word_found):
        return asp_word
    elif(adj_found):
        return 'Overall'
    else:
        asp_word = 'NOWORD'
        return asp_word


# There are 8113 files in Large Tripadvisor dataset
# It takes 7 hours to run the program with all the files
# k is the limited number of files
data_list=[]
k=0
# Collecting reviews for training
for infile in listing:
    curr_file = path+infile
    k+=1
    print(k)
    if k < 30 : # run for all- k>0 or to restrict ex. #if k < 2
        with open(curr_file) as data_file:
            data = json.load(data_file)
            reviews = data['Reviews']
            for rev in reviews:
                content = rev['Content']
                rating = rev['Ratings']
                aspect_terms_found = [ww for ww in rating] #terms given in the review
                sentences = tokenizer.tokenize(content)
                for s in sentences:
                    asp_word_found = extract_aspect_term(s)
                    data={}
                    if(asp_word_found!='NOWORD' and asp_word_found!='Overall' and asp_word_found in aspect_terms_found):
                        label=rating[asp_word_found]
                        data['text']=s
                        data['label'] = 'pos' if int(label) > senti_threshold else 'neg' #finds the pos or neg by comparing label(1-5) with senti_threshold
                        data_list.append(data)
                    elif (asp_word_found == 'Overall' and 'Overall' in aspect_terms_found): #this gets valid only when there is no term but their is an adjective
                        label = rating[asp_word_found]
                        data['text'] = s
                        data['label'] = 'pos' if float(label) > senti_threshold else 'neg'
                        data_list.append(data)


writeToJsonFile(writeFileName,data_list)

# aspect_terms_found=predefined_aspect_terms
# s="wireless were good"
# at=extract_aspect_term(s)
# print(at)