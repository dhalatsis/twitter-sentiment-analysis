#################
# Quick rundown of the sentiment analysis part of the pipeline
# Given a list of tweets with dates
# We assign a value in the interval [-1,1] that encodes the sentiment value
# This is currently down using the vaderSentiment library
# https://github.com/cjhutto/vaderSentiment
# Author D Halatsis


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd


def polarity_value_vader(tweets):
    SIA = SentimentIntensityAnalyzer()

    tweets['sent_value'] = tweets.text.apply(lambda x: SIA.polarity_scores(x))
    sentiment_value = tweets['sent_value'].apply(pd.Series)
    tweets_with_sentiment = pd.concat([tweets.drop(['sent_value'], axis=1), sentiment_value], axis=1)
    return tweets_with_sentiment

if __name__ == '__main__':
    import sys

    tweets_fname = sys.argv[1]
    tweets_sentiments_fname = sys.argv[2]

    tweets = pd.read_csv(tweets_fname, nrows=100000)
    # compute the score
    tweets_with_sentiment = polarity_value_vader(tweets)
    # save the file
    tweets_with_sentiment.to_csv(tweets_sentiments_fname)

