#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Title : Aspect Based Sentiment Analysis (Part 2 : Classifier)
# Authors : Kavya Danivas, Lilia Fkaier
# Dataset : Annotated Trip Advisor dataset http://nemis.isti.cnr.it/~marcheggiani/datasets/

import time
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords, words
import pickle
import json
import random
from nltk.classify import NaiveBayesClassifier

# User input paramters
traintestdevidefactor = 0.7 # 70% train_data 30% test_data
writeFileName = './trainDataVerylarge.json'  #train_data available AS sentence and polarity{text and label}
writeWordSetFileName = './wordSetData.json' #to store word_features

naiivepickleFileName = './naive_trained_pickle' #stores naiive bayes classifier pickle

label_cat = ['pos','neg']   #allowed polarities
stops=stopwords.words("english") #
stops.extend(['.', ',', "'s", "n't", '!', '(', ')', '-', ':', '!', '?', '...', '..', '+', ';', '<', '>'])
stop_words = set(stops)
a=set(['not','nor','no','aren','haven','isn','doesn', 'hasn', 'wasn', 'mustn', 'didnt', 'didn', 'shouldn', 'mightn','weren']) #whitelisted words
stop_words_modified=list(stop_words-a)  #final stop words
start_time = time.time()

'''
Function to write to a json file
'''
def writeToJsonFile(fileName, data_in):
    with open(fileName, 'w') as fp:
        json.dump(data_in, fp)


dataset = {}

with open(writeFileName, 'r') as fp: #opening writeFilename that consists of text and Label
    data = json.load(fp)
    document_tuple=[]
    all_words = []
    for label in label_cat: # loop through polarities,positive and negative
        tokenized_sentences = []
        for list_item in data: #loop through data (text,label)
            if (list_item['label']==label): #list_item['label'] will give either positive or negative
                s = list_item['text'] # text corresponding to label
                s=s.lower()
                tokenized_sentence_stop = nltk.word_tokenize(s)
                tokenized_sentence = [wd for wd in tokenized_sentence_stop if not wd in stop_words]
                all_words.extend(tokenized_sentence) #to create word features later
                document_tuple.append((tokenized_sentence, label)) #document tuple is a list of tuples(tokenized words,polarity). each line will have one seperate tuple inside the list
    random.shuffle(document_tuple)
    all_words=nltk.FreqDist(all_words)  #will give frequency distribution of all words
    word_features = list(all_words.keys())[:8000]  #word features contains top 8000 words in all_words
    dataset=word_features #eventually to store in a JSON file since word feature set is required while classifying any sentence in part 3(sentiment analysis)
    def find_features (document): #function to find features returns feature set
        words = set(document) #all the words in rev (see down below)
        features = {}
        for w in word_features: #loop through all the words in word feature
            features[w]=(w in words) # Right side returns boolean
        return features # keys are word features values are boolean

    featuresets = [(find_features(rev), label) for (rev, label) in document_tuple] #featuresets will be the input to classifier and must contain features and label
    traintestdevide=int(traintestdevidefactor*(len(featuresets))) #traintestdevide is the length of training_data
    print(len(featuresets))
    print(traintestdevide)
    training_set = featuresets[traintestdevide:] #first traintestdevide feature sets as training data
    testing_set = featuresets[:traintestdevide] #remaining as test_data
    # Naive bayes classifier
    classifier = nltk.NaiveBayesClassifier.train(training_set) #classifier

    print("Naive Bayes Algorithm Accuracy ", (nltk.classify.accuracy(classifier, testing_set))*100)  #to check accuracy
    classifier.show_most_informative_features(15) #returns top 15 features

    #to save classifier pickle
    save_classifier = open(naiivepickleFileName,"wb")
    pickle.dump(classifier, save_classifier)
    save_classifier.close()


writeToJsonFile(writeWordSetFileName,dataset)  #stores word features as we require that in sentiment analysis
