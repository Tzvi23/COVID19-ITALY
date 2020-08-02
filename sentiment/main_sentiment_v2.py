from sentiment import cnn_model3 as cnn
import numpy as np
from sentiment import twitter_scrap
import pandas as pd
import os


def sentiment_analysis(string, model, local_tokenizer):
    encoded_string = np.array([local_tokenizer.encode(string)])
    res = model.predict(encoded_string)
    if res > 0.5:
        print(f'{string} | Sentiment: Positive ({res})')
        return 1, res
    else:
        print(f'{string} | Sentiment: Negative ({res})')
        return 0, res


def load_variables(model_path, tokenizer_path):
    model = cnn.load_model(model_path)
    local_tokenizer = cnn.tokenizer_loadCreate(tokenizer_path)
    return model, local_tokenizer


def main_data(data_set_dir, sent_dir):
    model, tokenizer = load_variables('/Users/tzvip/PycharmProjects/COVID19-SIR/sentiment/cnn_model.h5', '/Users/tzvip/PycharmProjects/COVID19-SIR/sentiment/output')

    for data_file in os.listdir(data_set_dir):  # iterate over files in data set folder
        processed = list()
        cur_data = pd.read_csv(os.path.join(data_set_dir, data_file))  # Read data set file
        cur_data.drop_duplicates(subset='tweet', keep='first', inplace=True)  # Remove duplicated tweets (it happens)
        for index, row in cur_data.iterrows():  # Iterate over rows in df
            clean_tweet = cnn.clean_tweet(row['tweet'])  # Clean tweet
            ver, score = sentiment_analysis(clean_tweet, model, tokenizer)  # Classify the tweet
            cur_tweet = (row['tweet'], clean_tweet, ver, float(score[0][0]), row['date'], row['likes_count'])  # Create tuple for data frame
            processed.append(cur_tweet)

        # Create data frame for the results
        df = pd.DataFrame(processed, columns=['original_tweet', 'clean_tweet', 'sentiment', 'score', 'date', 'likes_count'])
        print(df.head())
        # Export data frame to csv
        df.to_csv(os.path.join(sent_dir, data_file), index=False)


main_data(data_set_dir='data_set_build/data_set_tweets',
          sent_dir='sentiment_processed')
