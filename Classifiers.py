import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import sqlite3

import time

# find word features for a single tweet
def word_feats(words):
	return dict([(word, True) for word in words])

# classifier
def naiveBayes(features_train, features_test):
	print 'train on %d instances, test on %d instances' % (len(features_train), len(features_test))
	classifier = NaiveBayesClassifier.train(features_train)
	print 'accuracy:', nltk.classify.util.accuracy(classifier, features_test)
	classifier.show_most_informative_features()	

# make features for traning set and testing set
def features_maker(class1, class2, className1, className2):
    class1feats = [(word_feats(words), className1) for words in class1]
    class2feats = [(word_feats(words), className2) for words in class2]
    class1cutoff = len(class1feats)*4/5
    class2cutoff = len(class2feats)*4/5
    trainfeats = class1feats[:class1cutoff] + class2feats[:class2cutoff]
    testfeats = class1feats[class1cutoff:] + class2feats[class2cutoff:]	
    return trainfeats, testfeats
