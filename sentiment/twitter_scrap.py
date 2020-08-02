import twint


def querySearch(query, lang, tweets_limit, start_date, end_date, dbName=None, colName=None):
    tweets = list()

    if start_date == 'YYYY-MM-DD':
        start_date = None
    if end_date == 'YYYY-MM-DD':
        end_date = None

    # Configure
    c = twint.Config()
    c.Search = query
    c.Lang = lang
    c.Limit = tweets_limit
    c.Store_object = True
    c.Since = start_date
    c.Until = end_date

    c.Store_object_tweets_list = tweets  # var to save into the tweets
    # Run
    twint.run.Search(c)
    print(tweets)

    return tweets
