#################
# Quick rundown of the sentiment analysis part of the pipeline
# Given a list of tweets with dates
# We assign a value in the interval [-1,1] that encodes the sentiment value
# This is currently down using the vaderSentiment library
# https://github.com/cjhutto/vaderSentiment
# Author D Halatsis


from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import numpy as np


def process_price_timeseries(btc_prices, start_date, end_date, window):
    btc_prices['open_time'] = pd.to_datetime(btc_prices['open_time'], format='%Y-%m-%d %H:%M:%S')
    btc_prices_2 = btc_prices.resample('60min', on='open_time').mean()
    btc_ts = btc_prices[
        (btc_prices['open_time'] > pd.to_datetime(start_date)) & (btc_prices['open_time'] < pd.to_datetime(end_date))]

    # drop and resample every day
    btc_ts.reset_index().drop('index', axis=1)
    btc_ts_2 = btc_ts.resample('D', on='open_time').mean()
    btc_ts_2['price'] = btc_ts_2['open'].rolling(window=5).mean()

    # delta function
    btc_ts_2['delta'] = btc_ts_2['price'] - btc_ts_2['open'].shift(-1)

    # clear up
    ts = btc_ts_2.reset_index()
    ts['id'] = ts.index
    ts = ts[['id', 'price', 'open_time']]
    return ts


def find_minima_maxima(ts):
    from scipy.signal import argrelextrema
    import numpy as np
    n = 5

    ts['min'] = ts.iloc[argrelextrema(ts.price.values, np.less_equal,
                                      order=n)[0]]['price']
    ts['max'] = ts.iloc[argrelextrema(ts.price.values, np.greater_equal,
                                      order=n)[0]]['price']


def simplify_and_correct(ts):
    ts_corrected = ts.fillna(ts.iloc[4].price)
    ts = ts_corrected.set_index('open_time').drop(columns=['id', 'min', 'max'])
    ts = ts.price.apply(lambda x: np.log2(x + 1))


# note here that start and en
def filter_period(df, name,start, end):
    df = df[(df[name] > pd.to_datetime(start)) & (df[name] < end)]
    return df


def parse_datetimes(column):
    return pd.to_datetime(column, format='%Y-%m-%d %H:%M:%S')


def pre_process_twitter_data(tweet_sentiments, start_date, end_date, frequency, logscale=True):
    tweet_info = tweet_sentiments[['date', 'compound']].sort_values(by='date')
    tweet_info.date = tweet_info.date.apply(lambda x: x.floor(frequency))
    # compute mean by day
    mean_compound_by_date = tweet_info.groupby(['date']).mean()

    # reset
    mean_compound_by_date = mean_compound_by_date.reset_index()
    # apply logscale
    if logscale:
        mean_compound_by_date.compound = mean_compound_by_date.compound.apply(lambda x: np.log2(x + 1))
    # fill in missing days to create a complete timeline
    mean_compound_by_date.index = pd.DatetimeIndex(mean_compound_by_date.date)
    mean_compound_by_date = mean_compound_by_date.drop('date', axis=1)
    mean_compound_by_date = mean_compound_by_date.asfreq(frequency).fillna(0)
    mean_compound_by_date = mean_compound_by_date[start_date:end_date]

    return mean_compound_by_date

def polarity_value_vader(tweets):
    SIA = SentimentIntensityAnalyzer()
    tweets.text = tweets.text.astype(str)
    tweets['sent_value'] = tweets.text.apply(lambda x: SIA.polarity_scores(x))
    sentiment_value = tweets['sent_value'].apply(pd.Series)
    tweets_with_sentiment = pd.concat([tweets.drop(['sent_value'], axis=1), sentiment_value], axis=1)
    return tweets_with_sentiment

# this function as seen here parses the date into python datetime so it can be put on a plot
def parse_date(tweet_sentiments, name='date'):
    tweet_sentiments[name] = pd.to_datetime(tweet_sentiments[name], format='%Y-%m-%d %H:%M:%S', errors='coerce')
    tweet_sentiments.sort_values(by=[name])

    # plt.plot(tweet_sentiments['compound'])
    tweet_sentiments

    tweet_sentiments.shape

    tweet_sentiments[name] = pd.to_datetime(tweet_sentiments[name], format='%Y-%m-%d %H:%M:%S', errors='coerce')

    tweet_sentiments = tweet_sentiments[tweet_sentiments[name].apply(lambda x: x != pd.NaT)]

    tweet_sentiments.sort_values(by=[name])

    return tweet_sentiments

if __name__ == '__main__':
    import sys

    tweets_fname = sys.argv[1]
    tweets_sentiments_fname = sys.argv[2]
    from time import time
    t = time()
    # fix separator for different files
    tweets = pd.read_json(tweets_fname)
    print('time elapsed: {time}'.format(time = (time() - t)))
    # compute the score
    tweets_with_sentiment = polarity_value_vader(tweets)
    # save the file
    print('time elapsed: {time}'.format(time = (time() - t)))

    tweets_with_sentiment.to_csv(tweets_sentiments_fname)
    print('time elapsed: {time}'.format(time = (time() - t)))

