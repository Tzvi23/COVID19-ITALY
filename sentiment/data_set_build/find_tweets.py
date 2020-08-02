import twint
import pandas as pd
import os


def query_data():
    # Load form excel file the geo locations and events
    query_excel = '/Users/tzvip/PycharmProjects/COVID19-SIR/sentiment/data_set_build/italy_timeline.xlsx'
    query_df = pd.read_excel(open(query_excel, 'rb'), sheet_name='search_queries')

    print(query_df.head())
    # Convert to dict format
    data_query = query_df.to_dict('split')['data']

    query_list = list()

    # Create from each line a query dict repr
    def create_query_dict(q_list, data_query, query_string):
        for q in data_query:
            print(q)
            single_query = dict()
            date = q[0].date().strftime("%Y-%m-%d")
            single_query['query_search'] = query_string
            single_query['date'] = '2020-01-31'  # can be replaced with date
            single_query['geo'] = q[2] + ',' + q[3]
            single_query['fileName'] = q[4] + '_' + q[1] + '_' + date.replace('/', '_') + '.csv'
            q_list.append(single_query)

    create_query_dict(query_list, data_query, 'coronavirus')
    print(query_list)

    # Create the twint config
    save_dir_path = 'data_set_tweets'
    if not os.path.exists(save_dir_path):
        os.mkdir(save_dir_path)

    tweet_config = twint.Config()
    for query in query_list:
        # Configure
        tweet_config.Search = query['query_search']
        tweet_config.Lang = 'en'
        tweet_config.Since = query['date']
        tweet_config.Until = None
        tweet_config.Geo = query['geo']
        # Storage
        tweet_config.Store_csv = True
        tweet_config.Output = os.path.join(save_dir_path, query['fileName'])

        twint.run.Search(tweet_config)


def delete_duplicates(file_path):
    df = pd.read_csv(file_path)
    df.drop_duplicates(subset='tweet', keep='first', inplace=True)
    return df


# test = delete_duplicates('/Users/tzvip/PycharmProjects/COVID19-SIR/sentiment/data_set_build/data_set_tweets/Event_Rome_2020-02-22.csv')
# print(test)

