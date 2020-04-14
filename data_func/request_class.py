import requests
import datetime
import os
import csv
import pandas as pd
import pickle


def get_time():
    date = datetime.datetime.now()
    return date.strftime('%x').replace('/', '_')


def load_obj(date, country=None):
    """
    Loads existing pickle file to obj.
    :param date: search by date string format: 'DD_MM_YY'
    :param country: If country supplied will look in the output folder for specific country dir
    :return: load pickle file to variable
    """
    # Define paths
    parent = os.path.sep.join(os.getcwd().split(os.path.sep)[:-1])  # Get parent folder
    output_path = os.path.join(os.path.join(parent, 'output'))
    history_path = os.path.join(output_path, 'all_history')
    # Load data
    if country is None:  # If country was not supplied means all history request
        for files in os.listdir(history_path):
            if files.replace('.pickle', '') == date:
                with open(os.path.join(history_path, files), 'rb') as load_file:
                    return pickle.load(load_file)
    else:
        for r, d, f in os.walk(output_path):
            print(f'root: {f}, dir: {d}, file:{f}')
            if (len(r) and country in r) and (len(f) and date + '.pickle' in f):
                with open(os.path.join(r, date + '.pickle'), 'rb') as load_file:
                    return pickle.load(load_file)
    print(f'{date}.pickle not found!')


class request:
    def __init__(self, name=get_time()):
        self.name = name
        self.base = 'https://corona.lmao.ninja/'  # API - endpoint
        self.timePeriod = None
        self.country = None
        self.URL = None
        self.response = None
        self.data = None
        self.df = None

    def create_historical_country_request(self, countryName, daysBack=30):
        """
        Creates a request for historical data for specific country
        for specific duration. Default set to 30 days
        """
        historical_country = f'v2/historical/{countryName}?lastdays={daysBack}'
        self.URL = self.base + historical_country
        self.timePeriod = daysBack
        self.country = countryName
        self.print_url()

    def create_historical_all_request(self, daysBack=30):
        """
        Creates a request for historical data for all countries for specific duration.
        Default set to 30 days
        """
        historical_all = f'v2/historical?lastdays={daysBack}'
        self.URL = self.base + historical_all
        self.timePeriod = daysBack
        self.print_url()

    def send_request(self):
        """
        Sends the request and stores the data in self.data as DICT
        """
        self.response = requests.get(url=self.URL)
        print(f'Request result: {self.response}')
        self.df = pd.read_json(self.response.text)
        self.data = self.response.json()

    def print_url(self):
        """
        Prints the url when the request is created
        """
        print(f'URL set to: {self.URL}')

    def output_data(self):
        parent = os.path.sep.join(os.getcwd().split(os.path.sep)[:-1])  # Get parent folder
        output_path_dir = os.path.join(os.path.join(parent, 'output'))  # ./data_func/output
        if not os.path.exists(output_path_dir):  # check if output folder exists
            os.mkdir(output_path_dir)  # If not create output folder
        country_dir = os.path.join(output_path_dir, self.data['country'])  # ./data_func/output/Israel
        if not os.path.exists(country_dir):  # check if country folder exists
            os.mkdir(country_dir)
        # timeline_dir = os.path.join(country_dir, 'timeline')  # ./data_func/output/Israel/timeline
        # if not os.path.exists(timeline_dir):
        #     os.mkdir(timeline_dir)

        for sir in self.data['timeline'].items():
            with open(os.path.join(country_dir, f'{sir[0]}_{self.name}.csv'), 'w') as output_file:
                header = ['date', 'value']
                dict_writer = csv.DictWriter(output_file, fieldnames=header)

                dict_writer.writeheader()
                for k, v in sir[1].items():
                    dict_writer.writerow({'date': k, 'value': v})

    def save_obj(self):
        """
        Saves the created class to pickle file named as the class name.
        The path is set to ./output/all_history as default.
        If country specified in the class the path is set ot ./output/{country_name}
        """
        parent = os.path.sep.join(os.getcwd().split(os.path.sep)[:-1])  # Get parent folder
        output_path_dir = os.path.join(os.path.join(parent, 'output'))  # ./data_func/output
        if not os.path.exists(output_path_dir):  # check if output folder exists
            os.mkdir(output_path_dir)  # If not create output folder
        history_path = os.path.join(output_path_dir, 'all_history')  # ./data_func/output/all_history
        if self.country is not None:
            history_path = os.path.join(output_path_dir, self.country)
        if not os.path.exists(history_path):
            os.mkdir(history_path)
        # Save pickle obj
        with open(os.path.join(history_path, self.name + '.pickle'), 'wb') as save_file:
            pickle.dump(self, save_file)
            print(f'Saved class: {self.name}.pickle to: {history_path}')
        # Save Data frame as csv
        with open(os.path.join(history_path, self.name + '.csv'), 'w') as save_file:
            self.export_csv(os.path.join(history_path, self.name + '.csv'))
            print(f'Saved class: {self.name}.csv to: {history_path}')

    def export_csv(self, path):
        if self.df is not None:
            self.df.to_csv(path, index=False, header=True)


# Example
# x = request()
# x.create_historical_all_request(90)
# x.send_request()
# x.save_obj()
# print(x)

# x = load_obj('04_13_20')
# print(x)