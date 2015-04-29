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
ObjectiveTweets = database.readDB("manual","Tweet","Score",1)
SubjectiveTweets = (database.readDB("manual","Tweet","Score",1) + database.readDB("Sheet1","Tweet","Score",-1))[:len(ObjectiveTweets)]


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
# Subjective Tweets:  238
# Objective Tweets:  238
# train on 380 instances, test on 96 instances
# accuracy: 0.5
# Most Informative Features
#                    store = None           Object : Subjec =      1.0 : 1.0
#                     good = True           Object : Subjec =      1.0 : 1.0
#                     damn = None           Object : Subjec =      1.0 : 1.0
#                  already = None           Object : Subjec =      1.0 : 1.0
#                 compared = True           Object : Subjec =      1.0 : 1.0
#                  testing = None           Object : Subjec =      1.0 : 1.0
#                      360 = None           Object : Subjec =      1.0 : 1.0
#                   kinect = True           Object : Subjec =      1.0 : 1.0
#                      kid = None           Object : Subjec =      1.0 : 1.0
#                     live = None           Object : Subjec =      1.0 : 1.0