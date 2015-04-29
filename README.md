# SentimentalAnalysis

4/29/2015 10:49am

Add two other classifiers in the Classifiers.py: decision tree and maximun entropy, but they are not performing as good as the naive bayes.

Also added a accuracy and fitness calculator in Classifiers.py


4/29/2015 9:35am

Updated the MainPN.py and MainOS.py to enable or disable the lemmatizer, stemer, synReplacer, bigram finder

Updated the Classifiers.py and DataProcessing.py to include synReplacer and bigram finder

Sujective and Objective accuracy: 0.686567164179
Negative and Possitive accuracy: 0.868421052632

4/29/2015 12:30am
test.db - the manually scored tweets

manual.xlsx - the manually scored tweets in an excel format

MainOS.py - the main file for running the naive bayes classifier on Subjective and Objective tweets.
It is not very accurate.

MainPN.py - the main file for running the naive bayes classifier on positive and negative tweets.
It has 76.3% of accuracy on 36 testing tweets, need more training and testing data.

DataProcessing.py, GetData.py, Classifiers.py contain the utilities used to finish the task. Hopefully they are more resuable than before.
