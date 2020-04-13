import requests
import datetime
import os
import csv
import pandas as pd


def get_time():
    date = datetime.datetime.now()
    return date.strftime('%x').replace('/', '_')


class request:
    def __init__(self, name=get_time()):
        self.name = name
        self.base = 'https://corona.lmao.ninja/'  # API - endpoint
        self.URL = None
        self.response = None
        self.data = None
        self.text = None

    def create_historical_country_request(self, countryName, daysBack=30):
        """
        Creates a request for historical data for specific country
        for specific duration. Default set to 30 days
        """
        historical_country = f'v2/historical/{countryName}?lastdays={daysBack}'
        self.URL = self.base + historical_country
        self.print_url()

    def create_historical_all_request(self, daysBack=30):
        """
        Creates a request for historical data for all countries for specific duration.
        Default set to 30 days
        """
        historical_all = f'v2/historical?lastdays={daysBack}'
        self.URL = self.base + historical_all
        self.print_url()

    def send_request(self):
        """
        Sends the request and stores the data in self.data as DICT
        """
        self.response = requests.get(url=self.URL)
        self.text = self.response.text
        print(f'Request result: {self.response}')
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
        timeline_dir = os.path.join(country_dir, 'timeline')  # ./data_func/output/Israel/timeline
        if not os.path.exists(timeline_dir):
            os.mkdir(timeline_dir)

        for sir in self.data['timeline'].items():
            with open(os.path.join(timeline_dir, f'{sir[0]}_{self.name}.csv'), 'w') as output_file:
                header = ['date', 'value']
                dict_writer = csv.DictWriter(output_file, fieldnames=header)

                dict_writer.writeheader()
                for k, v in sir[1].items():
                    dict_writer.writerow({'date': k, 'value': v})


# Example
x = request()
x.create_historical_all_request(90)
x.send_request()
df = pd.read_json(x.text)
print()
