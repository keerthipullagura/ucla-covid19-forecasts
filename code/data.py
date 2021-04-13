# import request
import pandas as pd
import numpy as np
import datetime
import warnings
import us

class Data(object):
    def date_range(self):
        warnings.warn('Data range method does not implement')
        raise NotImplementedError

    def get(self, start_date, end_date):
        warnings.warn('Data get method does not implement')
        raise NotImplementedError


'''
Inherit the Data Object:
class xxx(Data):
    def __init__(self):
        pass
    def range(self, start_date, end_date):
        pass
    def get(self, start_date, end_date):
        pass
'''


class NYTimes(Data):
    def __init__(self, level='states'):
        assert level == 'states' or level == 'counties', 'level must be [states|counties]'
        #url = 'https://raw.githubusercontent.com/nytimes/covid-19-data/master/us-' + level + '.csv'
        #self.table = pd.read_csv("http://raw.githubusercontent.com/nytimes/covid-19-data/master/us-counties.csv").drop('fips', axis=1)

        #http: // raw.githubusercontent.com / nytimes / covid - 19 - data / master / us - counties.csv
        # Can change this to pyspark DF.
        if level == 'states':
            self.table = pd.read_csv("fetchedData/us-states.csv").drop('fips', axis=1)
        elif level == 'counties':
            self.table = pd.read_csv("fetchedData/us-counties.csv").drop('fips', axis=1)
        self.level = level
        # Can change this to pyspark DF.
        self.state_list = self.table["state"].unique()

    def date_range(self, state, county=None):
        assert self.level == 'states' or county is not None, 'select a county for level=counties'
        state_table = self.table[self.table['state']
                                 == us.states.lookup(state).name]
        if self.level == 'states':
            tab = state_table
        else:
            tab = state_table[state_table['county'] == county]
        tab = tab.sort_values(by='date')
        date = tab['date'].unique()
        return date[0], date[-1]

    def get(self, start_date, end_date, state, county=None):

        assert self.level == 'states' or county is not None, 'select a county for level=counties'
        state_table = self.table[self.table['state']
                                 == us.states.lookup(state).name]
        if self.level == 'states':
            tab = state_table
        else:
            tab = state_table[state_table['county'] == county]
        date = pd.to_datetime(tab['date'])
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        # print(end_date)
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        mask = (date >= start) & (date <= end)
        return tab[mask]['cases'].to_numpy(), tab[mask]['deaths'].to_numpy()