import tweepy
import pandas as pd
import time
import re
from textblob import TextBlob
from textblob import Blobber
from textblob.sentiments import NaiveBayesAnalyzer
from nltk.corpus import stopwords
from collections import Counter
import string
import datetime
# ...

tb = Blobber(analyzer=NaiveBayesAnalyzer())

'''
    Utility function to clean tweet text by removing links, special characters
    using simple regex statements.
'''
def clean_tweet(tweet):
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t]) | (\w +:\ / \ / \S +)", " ", tweet).split())


'''
    Utility function to classify sentiment of passed tweet using textblob's sentiment method. 
    Include Pattern Analysis and Naives Bayes Analysis
'''
def get_tweet_sentiment(tweet):
    pattern_analysis = TextBlob(tweet)
    naives_analysis = tb(tweet)
    return [pattern_analysis.sentiment, naives_analysis.sentiment]


'''
    Save a dataframe to csv format.
'''
def saveCSV(df, fileName):
    fd = open(fileName, 'w+')
    df.to_csv(fd, index=True, encoding='utf-8', header=True)
    fd.close()


'''
    Get a list of the most common words in a collection of tweets.
'''
def get_most_common_words(df, searchValue, amount):
    word_list = pd.Series(' '.join(df['text']).lower().split())
    filtered_words = [word for word in word_list if word not in stopwords.words('english')]
    word_dict = Counter(filtered_words)
    for k, v in word_dict.items():
        if k in searchValue.lower().split() or k == "rt" or k in string.punctuation:
            del word_dict[k]
    return word_dict.most_common(amount)


'''
    Add a row to a dataframe.
'''
def addRowToDF(df, arrayVal):
    df.loc[-1] = arrayVal  # adding a row
    df.index = df.index + 1  # shifting index
    df = df.sort_index()  # sorting by index
    return df


'''
    Analyze the twitter sentiment of a given search value.
'''
def analyze_twitter_sentiment(searchValue):
    print("Analyzing twitter sentiment for "+searchValue+" ...")
    ACCESS_TOKEN = '355258725-ieNBIlHNqMM2KpdDZUbYDxDeGfNqQv6uclV5xb7l'
    ACCESS_SECRET = 'GQI4XFLLmcytfoOTV3BMQalyR583Z1gGDpwEE6nYQdehl'
    CONSUMER_KEY = 'Tg2tQvfjYYB5wj5d1ib8mdQfU'
    CONSUMER_SECRET = 'b47muF1CveaLrVzLFbDCSEUuHwRkQT9zuZww64TCnCwoaou3FT'

    # authorize twitter, initialize tweepy
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)
    api = tweepy.API(auth)


    # make initial request for most recent tweets (200 is the maximum allowed count)
    new_tweets = api.search(q=searchValue, count=100)

    tweets =[]

    # iterate through the tweets and construct a dataframe of the significant results.
    for tweet in new_tweets:

        tweetText = clean_tweet(tweet.text)
        tweetRetweetCount = tweet.retweet_count
        tweetCreatedAt = tweet.created_at
        tweetUsername = tweet.user.screen_name
        tweetFollowersCount = tweet.user.followers_count
        sentiment = get_tweet_sentiment(tweetText);

        tweets.append(
            {
                "username": tweetUsername,
                "text": clean_tweet(tweetText).encode('utf-8'),
                "created_at": tweetCreatedAt,
                "retweet_count": tweetRetweetCount,
                "followers_count": tweetFollowersCount,
                "pattern-polarity": sentiment[0].polarity,
                "pattern-subjectivity": sentiment[0].subjectivity,
                "naivesbayes-probability-positive": sentiment[1].p_pos,
                "naivesbayes-probability-negative": sentiment[1].p_neg
            }
        )

    df = pd.DataFrame(pd.DataFrame.from_dict(tweets, orient='columns'))
    #df['searchvalue'] = searchValue.encode('utf-8')

    #print(df["sentiment"].mean())
    #saveCSV(df, '../results/'+searchValue+'-rawsentimentdata.csv')

    return df


'''
    Analyze the sentiment of each term in a list of search values
'''
def analyze_list(inputFile):
    df_allTerms = pd.DataFrame(columns=["term", "entrydate", "num_of_tweets", "pattern_polarity_mean",
                                        "pattern_subjectivity_mean", "naivesbayes_p_pos_mean", "naivesbayes_p_neg_mean",
                                        "pattern_max_tweet", "naives_max_tweet", "most_common_words"])
    file = open(inputFile, "r")
    for searchTerm in file.read().splitlines():
        df_rawTwitterData = analyze_twitter_sentiment(searchTerm)

        if df_rawTwitterData.empty:
            print("empty dataframe.")
        else:

            pattern_max_id = df_rawTwitterData['pattern-polarity'].idxmax()
            naives_max_id = df_rawTwitterData['naivesbayes-probability-positive'].idxmax()

            if pattern_max_id != 0:
                pattern_max_tweet = df_rawTwitterData['text'].values[pattern_max_id]
            else:
                pattern_max_tweet = ""

            if naives_max_id != 0:
                naives_max_tweet = df_rawTwitterData['text'].values[naives_max_id]
            else:
                naives_max_tweet = ""

            num_of_tweets = df_rawTwitterData.shape[0]
            pattern_polarity_mean = df_rawTwitterData["pattern-polarity"].mean()
            pattern_subjectivity_mean = df_rawTwitterData["pattern-subjectivity"].mean()
            naives_positive_mean = df_rawTwitterData["naivesbayes-probability-positive"].mean()
            naives_negative_mean = df_rawTwitterData["naivesbayes-probability-negative"].mean()
            commons = get_most_common_words(df_rawTwitterData, searchTerm, 10)
            df_allTerms = addRowToDF(df_allTerms, [searchTerm, datetime.datetime.now(), num_of_tweets,
                                                   pattern_polarity_mean, pattern_subjectivity_mean,
                                                   naives_positive_mean, naives_negative_mean, pattern_max_tweet,
                                                   naives_max_tweet, commons])

    saveCSV(df_allTerms, "results/final-twitter-data.csv")


#########################################################################

#analyze_twitter_sentiment() <-- single value

analyze_list("source.txt")