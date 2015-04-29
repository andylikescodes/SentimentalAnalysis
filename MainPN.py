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
from Classifiers import decisionTree
from Classifiers import axentClassifier


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
negativeTweets = processor.processTweet(negativeTweets, 
										lemmatize=False, 
										stem=False, 
										synReplace=True)
possitiveTweets = processor.processTweet(possitiveTweets,
										lemmatize=False, 
										stem=False, 
										synReplace=True)



# make features
train, test = features_maker(negativeTweets, 
							possitiveTweets, 
							'Negative', 
							'Positive', 
							bigram=True)

# train and test the classifier
naiveBayes(train, test)
# decisionTree(train, test)
# axentClassifier(train, test)

###################################################################
# lematizer: off
# stemmer: off
# synReplacer: on
# bigram: on
# possitive Tweets:  94
# negative Tweets:  94
# train on 150 instances, test on 38 instances
# accuracy: 0.868421052632
# {'Positive': 0.85, 'Negative': 0.8888888888888888} {'Positive': 0.89473684210526
# 32, 'Negative': 0.8421052631578947}
# Most Informative Features
#                  repulse = True           Negati : Positi =      5.7 : 1.0
#                    sound = True           Negati : Positi =      4.3 : 1.0
#                  picture = True           Positi : Negati =      3.8 : 1.0
#                     plug = True           Positi : Negati =      3.7 : 1.0
#                    would = True           Negati : Positi =      3.7 : 1.0
#                microsoft = True           Negati : Positi =      3.4 : 1.0
#                     shop = True           Negati : Positi =      3.0 : 1.0
#     phonograph_recording = True           Negati : Positi =      3.0 : 1.0
# (u'phonograph_recording', u'repulse') = True           Negati : Positi =      3.
# 0 : 1.0
#                 upwardly = True           Positi : Negati =      3.0 : 1.0

###################################################################
# lematizer: on
# stemmer: off
# synReplacer: on
# bigram: on
# accuracy: 0.815789473684
# Most Informative Features
#                  repulse = True           Negati : Positi =      5.7 : 1.0
#                  picture = True           Positi : Negati =      3.8 : 1.0
#                     plug = True           Positi : Negati =      3.7 : 1.0
#                    sound = True           Negati : Positi =      3.7 : 1.0
#                    would = True           Negati : Positi =      3.7 : 1.0
#                microsoft = True           Negati : Positi =      3.4 : 1.0
#                     shop = True           Negati : Positi =      3.0 : 1.0
#     phonograph_recording = True           Negati : Positi =      3.0 : 1.0
# (u'phonograph_recording', u'repulse') = True           Negati : Positi =      3.
# 0 : 1.0
#                 upwardly = True           Positi : Negati =      3.0 : 1.0

###################################################################
# lematizer: on
# stemmer: on
# synReplacer: on
# bigram: on
# accuracy: 0.789473684211
# Most Informative Features
#                  repulse = True           Negati : Positi =      5.7 : 1.0
#                  picture = True           Positi : Negati =      3.8 : 1.0
#                    would = True           Negati : Positi =      3.7 : 1.0
#                     chew = True           Positi : Negati =      3.7 : 1.0
#                microsoft = True           Negati : Positi =      3.4 : 1.0
#             the_likes_of = True           Positi : Negati =      3.2 : 1.0
#                     tone = True           Positi : Negati =      3.0 : 1.0
#                    excit = True           Positi : Negati =      3.0 : 1.0
#     phonograph_recording = True           Negati : Positi =      3.0 : 1.0
# (u'phonograph_recording', u'repulse') = True           Negati : Positi =      3.
# 0 : 1.0


##################################################################
# lematizer: on
# stemmer: on
# synReplacer: on
# bigram: off
# possitive Tweets:  94
# negative Tweets:  94
# train on 150 instances, test on 38 instances
# accuracy: 0.763157894737
# Most Informative Features
#                  repulse = True           Negati : Positi =      5.7 : 1.0
#                  picture = True           Positi : Negati =      3.8 : 1.0
#                    would = True           Negati : Positi =      3.7 : 1.0
#                     chew = True           Positi : Negati =      3.7 : 1.0
#                microsoft = True           Negati : Positi =      3.4 : 1.0
#             the_likes_of = True           Positi : Negati =      3.2 : 1.0
#                     tone = True           Positi : Negati =      3.0 : 1.0
#           limited_review = True           Positi : Negati =      3.0 : 1.0
#     phonograph_recording = True           Negati : Positi =      3.0 : 1.0
#            pauperization = True           Positi : Negati =      3.0 : 1.0


##################################################################
# lematizer: on
# stemmer: on
# synReplacer: off
# bigram: off
# possitive Tweets:  94
# negative Tweets:  94
# train on 150 instances, test on 38 instances
# accuracy: 0.736842105263
# Most Informative Features
#                    drive = True           Negati : Positi =      5.7 : 1.0
#                    video = True           Positi : Negati =      3.8 : 1.0
#                     plug = True           Positi : Negati =      3.7 : 1.0
#                    would = True           Negati : Positi =      3.7 : 1.0
#                microsoft = True           Negati : Positi =      3.4 : 1.0
#                     like = True           Positi : Negati =      3.2 : 1.0
#                     look = True           Positi : Negati =      3.0 : 1.0
#                     disc = True           Negati : Positi =      3.0 : 1.0
#                   awesom = True           Positi : Negati =      3.0 : 1.0
#                   review = True           Positi : Negati =      3.0 : 1.0

#################################################################
# lematizer: on
# stemmer: off
# synReplacer: off
# bigram: off
# possitive Tweets:  94
# negative Tweets:  94
# train on 150 instances, test on 38 instances
# accuracy: 0.736842105263
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

#################################################################
# test result with bigram and synonym replacement
# possitive Tweets:  94
# negative Tweets:  94
# train on 150 instances, test on 38 instances
# accuracy: 0.815789473684
# Most Informative Features
#                  repulse = True           Negati : Positi =      5.7 : 1.0
#                  picture = True           Positi : Negati =      3.8 : 1.0
#                     plug = True           Positi : Negati =      3.7 : 1.0
#                    sound = True           Negati : Positi =      3.7 : 1.0
#                    would = True           Negati : Positi =      3.7 : 1.0
#                microsoft = True           Negati : Positi =      3.4 : 1.0
#                     shop = True           Negati : Positi =      3.0 : 1.0
#     phonograph_recording = True           Negati : Positi =      3.0 : 1.0
# (u'phonograph_recording', u'repulse') = True           Negati : Positi =      3.
# 0 : 1.0
#                 upwardly = True           Positi : Negati =      3.0 : 1.0

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