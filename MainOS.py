#####################################################################
#
#this is the main for classifying the positive and negative tweets
#
################################################################

from GetData import Database
from DataProcessing import DataProcessor
from Classifiers import naiveBayes
from Classifiers import features_maker

# Initialize the database object
database = Database("test.db")

# Initiatize the data processor object
processor = DataProcessor()

# Read original tweets from database and make them having the same number of tweets, ready for process

SubjectiveTweets = (database.readDB("manual","Tweet","Score",1) + database.readDB("manual","Tweet","Score",-1))
ObjectiveTweets = database.readDB("manual","Tweet","Score",0)[:len(SubjectiveTweets)]

print "Subjective Tweets: ", len(SubjectiveTweets)
print "Objective Tweets: ", len(ObjectiveTweets)

# Use the processor to processor the tweets, ready for making features
SubjectiveTweets = processor.processTweet(SubjectiveTweets)
ObjectiveTweets = processor.processTweet(ObjectiveTweets)

# make features
train, test = features_maker(SubjectiveTweets, ObjectiveTweets, 'Subjective', 'Objective')

# train and test the classifier
naiveBayes(train, test)

###############################################################
# Subjective Tweets:  332
# Objective Tweets:  332
# train on 530 instances, test on 134 instances
# accuracy: 0.611940298507
# Most Informative Features
#              playstation = True           Object : Subjec =     12.3 : 1.0
#           forzadiscovery = True           Subjec : Object =      6.3 : 1.0
#                 employee = True           Object : Subjec =      5.7 : 1.0
#                     best = True           Subjec : Object =      5.7 : 1.0
#               controller = True           Object : Subjec =      5.7 : 1.0
#                    night = True           Subjec : Object =      5.0 : 1.0
#                     nice = True           Subjec : Object =      4.3 : 1.0
#                 actually = True           Subjec : Object =      4.3 : 1.0
#                     next = True           Subjec : Object =      3.8 : 1.0
#                    right = True           Object : Subjec =      3.8 : 1.0