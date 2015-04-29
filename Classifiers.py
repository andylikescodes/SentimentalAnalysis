import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews
import sqlite3
from DataProcessing import DataProcessor
from nltk.classify import DecisionTreeClassifier
from nltk.classify import MaxentClassifier
import collections
from nltk import metrics
# from nltk.classify.scikitlearn import SklearnClassifier
# from sklearn import MultinomialNB

import time

processor = DataProcessor()


# calculate the precision(accuracy) and the recall(fitness) of the model
def precision_recall(classifier, testfeats):
	refsets = collections.defaultdict(set)
	testsets = collections.defaultdict(set)

	for i, (feats, label) in enumerate(testfeats):
		refsets[label].add(i)
		observed = classifier.classify(feats)
		testsets[observed].add(i)

	precisions={}
	recalls={}

	for label in classifier.labels():
		precisions[label] = metrics.precision(refsets[label], testsets[label])
		recalls[label] = metrics.recall(refsets[label], testsets[label])

	return precisions, recalls


# find word features for a single tweet
def word_feats(words, bigram=False):
	if bigram==True:
		words = words + processor.findBigrams(words)
		return dict([(word, True) for word in words])
	else:
		return dict([(word, True) for word in words])

# classifier
def naiveBayes(features_train, features_test):
	print 'train on %d instances, test on %d instances' % (len(features_train), len(features_test))
	classifier = NaiveBayesClassifier.train(features_train)
	print 'accuracy:', nltk.classify.util.accuracy(classifier, features_test)
	classifier.show_most_informative_features()	
	precisions, recalls = precision_recall(classifier, features_test)
	print "accuracy: ", precisions, "fitness: ", recalls

# make features for traning set and testing set
def features_maker(class1, class2, className1, className2, bigram=False):
    class1feats = [(word_feats(words, bigram), className1) for words in class1]
    class2feats = [(word_feats(words, bigram), className2) for words in class2]
    class1cutoff = len(class1feats)*4/5
    class2cutoff = len(class2feats)*4/5
    trainfeats = class1feats[:class1cutoff] + class2feats[:class2cutoff]
    testfeats = class1feats[class1cutoff:] + class2feats[class2cutoff:]	
    return trainfeats, testfeats

# decision tree classifier
def decisionTree(features_train, features_test):
	print 'train on %d instances, test on %d instances' % (len(features_train), len(features_test))
	classifier = DecisionTreeClassifier.train(features_train,
												binary=True,
												entropy_cutoff=0.8,
												depth_cutoff=5,
												support_cutoff=30)
	print 'accuracy:', nltk.classify.util.accuracy(classifier, features_test)
	precisions, recalls = precision_recall(classifier, features_test)
	print "accuracy: ", precisions, "fitness: ", recalls

# maximum entropy classifier
def axentClassifier(features_train, features_test):
	print 'train on %d instances, test on %d instances' % (len(features_train), len(features_test))
	classifier = MaxentClassifier.train(features_train,algorithm='gis')
	print 'accuracy:', nltk.classify.util.accuracy(classifier, features_test)
	precisions, recalls = precision_recall(classifier, features_test)
	print "accuracy: ", precisions, "fitness: ", recalls

# def sklearnMultinomialNB(features_train, features_test):
# 	print 'train on %d instances, test on %d instances' % (len(features_train), len(features_test))
# 	classifier = SklearnClassifier(MultinomialNB())
# 	classifier.train
# 	print 'accuracy:', nltk.classify.util.accuracy(classifier, features_test)