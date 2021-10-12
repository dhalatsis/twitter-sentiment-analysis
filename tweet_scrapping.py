# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import pandas
# twitterscrapper didn't work because of an outdated proxy list
# from twitterscraper import query_tweets
import snscrape.modules.twitter as twitter # scraping module
import pandas as pd
import itertools
import datetime
from time import time
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer



def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.

def scrape(date):
	# creating generator for scraping
    tweets = twitter.TwitterSearchScraper(f"bitcoin since:{yest2str(date)} until:{date} filter:has_engagement lang:en").get_items()
    # iterating through all tweets
    tweets = itertools.islice(tweets, None)
    # storing tweets in pandas dataframe∂
    df = pd.DataFrame(tweets)
    # returning necessay columns of dataframe
    return df
    #return df[['date','content']]

yest2str = lambda date: datetime.datetime.strftime(datetime.datetime.strptime(date, "%Y-%m-%d") + datetime.timedelta(days=-10), "%Y-%m-%d")

if __name__ == '__main__':
    #list_of_tweets = query_tweets("Trump ", 10)

    # print the retrieved tweets to the screen:
    date = "2021-08-01"

    s = time()
    # scraping for date
    # df = scrape(date)
    # e = time()
    # print("Done")
    # # creating dataframe that stores the date that has been scraped, the number of iterations, the time that the scraping occured and the duration of the scraping
    # dft = pd.DataFrame({"date": yest2str(date), "iter": f"{1}",
    #                     "time": datetime.datetime.strftime(datetime.datetime.now(), "%d-%m %H:%M:%S"),
    #                     "duration": [e - s]})
    # # appending infos dataframe to csv file
    # dft.to_csv("history.csv", mode="a", index=False, header=False)
    # # appending tweets dataframe to csv file
    # df.to_csv("tweets_4.csv", mode="a", index=False, header=False)
    # print(e-s)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
