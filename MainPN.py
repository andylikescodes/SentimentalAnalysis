#####################################################################
#
#this is the main for classifying the positive and negative tweets
#
################################################################

from GetData import Database
from DataProcessing import DataProcessor
from Classifiers import naiveBayes
from Classifiers import features_maker
from nltk.corpus import stopwords

# Initialize the database object
database = Database("test.db")
# Initiatize the data processor object
processor = DataProcessor()

# Read original tweets from database and make them having the same number of tweets, ready for process
negativeTweets = database.readDB("manual","Tweet","Score",-1)
possitiveTweets = database.readDB("manual","Tweet","Score",1)[:len(negativeTweets)]

print "possitive Tweets: ", len(possitiveTweets)
print "negative Tweets: ", len(negativeTweets)

# Use the processor to processor the tweets, ready for making features
negativeTweets = processor.processTweet(negativeTweets)
possitiveTweets = processor.processTweet(possitiveTweets)

# make features
train, test = features_maker(negativeTweets, possitiveTweets, 'Negative', 'Positive')

# train and test the classifier
naiveBayes(train, test)

#####################################################################
# RESULT:
# possitive Tweets:  94
# negative Tweets:  94
# train on 150 instances, test on 38 instances
# accuracy: 0.763157894737
# Most Informative Features
#                    drive = True           Negati : Positi =      5.0 : 1.0
#                  getting = True           Negati : Positi =      5.0 : 1.0
#                    video = True           Positi : Negati =      3.8 : 1.0
#                  plugged = True           Positi : Negati =      3.7 : 1.0
#                    would = True           Negati : Positi =      3.7 : 1.0
#                microsoft = True           Negati : Positi =      3.4 : 1.0
#                     disc = True           Negati : Positi =      3.0 : 1.0
#                  excited = True           Positi : Negati =      3.0 : 1.0
#                    voice = True           Negati : Positi =      3.0 : 1.0
#                   review = True           Positi : Negati =      3.0 : 1.0