# -*- coding: utf-8 -*-
"""
Created on Mon Oct 24 12:47:41 2022

@author: mikaa
"""

#Importing necessary libraries

import snscrape.modules.twitter as sntwitter
from textblob import TextBlob
import sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import nltk
import re
import string
from PIL import Image
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.stem import SnowballStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('vader_lexicon')

#Search Amazon on Twitter

query = 'Amazon -is:retweet lang:en'
tweets = []
limit = 100
for tweet in sntwitter.TwitterSearchScraper(query).get_items():
    if len(tweets) == limit :
        break
    else : 
        tweets.append([tweet.date, tweet.user.username, tweet.content, tweet.lang, tweet.retweetCount, tweet.likeCount])
        
 #Creating a Dataframe with the tweets

df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet', 'Lang', 'Retweet Count', 'Like Count'])
df.head(10)
#Converting df to a csv file
df.to_csv('tweets.csv')

#Tweet cleaning
import numpy as np
import re

stopwords = ["for", "on", "an", "a", "of", "and", "in", "the", "to", "from"]

def clean_tweet(tweet):
    if type(tweet) == np.float:
        return ""
    temp = tweet.lower()
    temp = re.sub("'", "", temp) # to avoid removing contractions in english
    temp = re.sub("@[A-Za-z0-9_]+","", temp) #mentions
    temp = re.sub("#[A-Za-z0-9_]+","", temp) #Hastags
    temp = re.sub(r'http\S+', '', temp) #Links
    temp = re.sub('[()!?]', ' ', temp) #Punctuation
    temp = re.sub('\[.*?\]',' ', temp) #Punctuation
    temp = re.sub("[^a-z0-9]"," ", temp) #Non alpha-numeric characters
    temp = temp.split()
    temp = [w for w in temp if not w in stopwords]
    temp = " ".join(word for word in temp)
    return temp

df['Tweet'] = [clean_tweet(tw) for tw in df['Tweet']]
df['Tweet'].drop_duplicates(inplace = True)
df

nltk.download()



#Sentiment Analysis in 3 parts : Positive tweets, negative tweets and neutral tweets

positive = 0
negative = 0
neutral = 0
polarity = 0
tweet_list = []
neutral_list = []
negative_list = []
positive_list = []


for tweet in df['Tweet']:
    tweet_list.append(tweet)
    analysis = TextBlob(tweet)
    score = SentimentIntensityAnalyzer().polarity_scores(tweet)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    comp = score['compound']
    polarity += analysis.sentiment.polarity
    
    if neg > pos : #Append in negative tweets list
        negative_list.append(tweet)
        negative += 1
        
    elif pos > neg : #Append in positive tweets list
        positive_list.append(tweet)
        positive += 1
        
    elif pos == neg : #Append in neutral tweets list
        neutral_list.append(tweet)
        neutral +=1
        
        
        
def percentage(part,whole):
 return 100 * float(part)/float(whole)

positive = percentage(positive, limit)
negative = percentage(negative, limit)
neutral = percentage(neutral, limit)
polarity = percentage(polarity, limit)
positive = format(positive, '.1f')
negative = format(negative, '.1f')
neutral = format(neutral, '.1f')


#Creating DataFrames
tweet_list = pd.DataFrame(tweet_list)
neutral_list = pd.DataFrame(neutral_list)
negative_list = pd.DataFrame(negative_list)
positive_list = pd.DataFrame(positive_list)

#Number of Tweets (Total, Positive, Negative, Neutral)
print("Total tweet number: ",len(tweet_list))
print("Positive tweets number: ",len(positive_list))
print("Negative tweets number: ", len(negative_list))
print("Neutral tweets number: ",len(neutral_list))

#Creating PieCart
import streamlit as st 

labels = ['Positive ['+str(positive)+'%]' , 'Neutral ['+str(neutral)+'%]','Negative ['+str(negative)+'%]']
sizes = [positive, neutral, negative]
colors = ['limegreen', 'blue','red']
#patches, texts = plt.pie(sizes,colors=colors, startangle=90)
#plt.style.use('default')
#plt.legend(labels)
#plt.title("Twitter Sentiment Analysis Result for Amazon" )
#plt.axis('equal')
#st.pyplot(patches)

explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.


st.title("Amazon's reputation on Twitter in 2022")
st.pyplot(fig1)
