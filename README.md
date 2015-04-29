# SentimentalAnalysis

4/29/2015 12:30am
Updated the MainPN.py and MainOS.py to enable or disable the lemmatizer, stemer, synReplacer, bigram finder

Updated the Classifiers.py and DataProcessing.py to include synReplacer and bigram finder

4/29/2015 12:30am
test.db - the manually scored tweets

manual.xlsx - the manually scored tweets in an excel format

MainOS.py - the main file for running the naive bayes classifier on Subjective and Objective tweets.
It is not very accurate.

MainPN.py - the main file for running the naive bayes classifier on positive and negative tweets.
It has 76.3% of accuracy on 36 testing tweets, need more training and testing data.

DataProcessing.py, GetData.py, Classifiers.py contain the utilities used to finish the task. Hopefully they are more resuable than before.
