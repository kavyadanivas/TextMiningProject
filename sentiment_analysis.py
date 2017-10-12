#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Title : Aspect Based Sentiment Analysis (Par 3 - aspect term and polaity detection)
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
import pickle
from nltk.classify import NaiveBayesClassifier

lemmatizer=WordNetLemmatizer()      #to give similar meaning of a word
tokenizer_reg = RegexpTokenizer(r'\w+') #to tokenize + removing punctuations
tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")

# User input paramters
sim_val_threshold = 0.4 # Similarity threshold
path = 'LargeTripadvisor/'
senti_threshold = 3 # between 1 to 5, if val > senti_threshold, then pos, else neg
outFileName = './absa_out.json'    #output is displayed on a file
WordSetFileName = './wordSetData.json' #word feature file
naiivepickleFileName = './naive_trained_pickle' #location of the classifier pickle

# Predefine aspect terms in the file and associated aspect terms considered in the project
predefined_aspects = {"Rooms" : "room", "Cleanliness": "cleanliness",
                      "Value":"value", "Service":"service",
                      "Location": "location", "Sleep Quality":"sleep",
                      "Business service (e.g., internet access)":"internet","Check in / front desk":"reception"}  #Business service (e.g., internet access)', 'Check in / front desk'

noun = ["NN", "NNS", "NNP", "NNPS"]             #defining POS tags for words #Currently not been used in the code
adjective = ["JJ", "JJR", "JJS"]                #defining POS tags for words #not used

stops=stopwords.words("english")
stops.extend(['.', ',', "'s", "n't", '!', '(', ')', '-', ':', '!', '?', '...', '..', '+', ';', '<', '>'])
stop_words = set(stops)
a=set(['not','nor','no','aren','haven','isn','doesn', 'hasn', 'wasn', 'mustn', 'didnt', 'didn', 'shouldn', 'mightn','weren'])
stop_words_modified=list(stop_words-a)
start_time = time.time()


# store aspect terms that are predefined
predefined_aspect_terms = predefined_aspects.keys() #aspect terms in the dataset
predef_aspect_term_modified=unique(predefined_aspects.values()) #aspect terms

# print aspect terms and categories that are predefined
print("Predefined Aspect Terms: ")
print(predefined_aspect_terms) #TODO: Clean the output

print("Defined Aspect Terms in the Project: ")
print(predef_aspect_term_modified) #TODO: Clean the output

start_time = time.time()
listing = os.listdir(path)

aspect_terms_found = []
word_features = []

with open(WordSetFileName, 'r') as f:
    word_features = json.load(f)

#Naive bayes classifier load
classifier_f = open(naiivepickleFileName,"rb")
classifier = pickle.load(classifier_f)
classifier_f.close()


'''
Function to find features
Input : Tokenized words
Output : features
'''
def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features


'''
Function to find polarity
Input : sentence
Output : polarity
'''
def findpolarity(sent):
    sent = sent.lower()
    tokenized_sentence_stop = nltk.word_tokenize(sent)
    tokenized_sentence = [wd for wd in tokenized_sentence_stop if not wd in stop_words]
    featureset=find_features(tokenized_sentence)
    label = nltk.NaiveBayesClassifier.classify(classifier, featureset)
    return  label


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
    print(filtered_review_line)
    max_sim_val = 0
    asp_word_found = False
    for w1 in filtered_review_line: # loop through all the pos tagged words
        for wn in predefined_aspect_terms: # in predefined aspect terms
            wnn=predefined_aspects[wn] # get the aspect term defined in the project
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

    if(asp_word_found): #checks if word token is in pre defined aspect term
        return asp_word
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
    if k < 3: # run for all k>0, to restrict, ex. #if k < 2)
        with open(curr_file) as data_file:
            data = json.load(data_file)
            reviews = data['Reviews']
            for rev in reviews:
                data = {}
                data_rev = {}
                content = rev['Content'] #customer review
                data["Content"]=content #to write in output file
                data_rev["Overall"]=findpolarity(content) # will be added in the output file under 'rating' key
                # rating = rev['Ratings']
                # aspect_terms_found = [ww for ww in rating]
                sentences = tokenizer.tokenize(content)
                for s in sentences:
                    polarity = findpolarity(s)
                    asp_word_found = extract_aspect_term(s)
                    if(asp_word_found!='NOWORD'): # checks whether aspect  term is found or not
                        data_rev[asp_word_found]=polarity #if found assign the polarity to the aspect term
                    print(s)
                    print(asp_word_found)
                    print(polarity)
                data["Ratings"]=data_rev  #assign data_rev containing aspect terms and polarity to the rating
                data_list.append(data) #append data containing content and rating to data list which will eventually be stored in absa_out file

writeToJsonFile(outFileName,data_list) #write data list to absa_out

# s="internet were good"
# at=extract_aspect_term(s)
# print(at)