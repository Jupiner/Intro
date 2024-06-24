import tweepy
from textblob import TextBlob
from twitter_api import consumer_key, consumer_secret, access_token, access_token_secret, bearer_token


client = tweepy.Client(bearer_token)

auth = tweepy.OAuth2AppHandler(API_key, API_secret)
# auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = client.search_recent_tweets('Trump')

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.tweet)
    print(analysis.sentiment)

