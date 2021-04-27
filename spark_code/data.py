import datetime
import warnings
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import DateType

#spark = SparkSession.builder.master("local[1]").appName("covid19seir").getOrCreate()
spark = SparkSession.builder.getOrCreate()

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

        if level == 'states':
            self.table = spark.read.format("csv").load("fetchedData/us-states.csv", format='csv', header='true', inferSchema='true').drop('fips')
        elif level == 'counties':
            self.table = spark.read.format("csv").load("fetchedData/us-counties.csv", format='csv', header='true', inferSchema='true').drop('fips')
        self.table = self.table.withColumn("casted_date",self.table['date'].cast(DateType()))
        self.level = level
        # Can change this to pyspark DF.
        '''ca_entry = self.table.filter(col('state') == 'California')
        #& col('casted_date') >=  datetime.datetime.strptime('2020-03-22', '%Y-%m-%d') & col('casted_date') <=  datetime.datetime.strptime('2020-05-30', '%Y-%m-%d'))
        if level == 'states':
            ca_entry.write.option("header", True).csv("ca_state")
        elif level == 'counties':
            ca_entry.write.option("header", True).csv("ca_counties")
        print('Cal from x-y', ca_entry.take(5))'''
        self.state_list = np.array(self.table.select('state').distinct().collect())

    def date_range(self, state, county=None):
        assert self.level == 'states' or county is not None, 'select a county for level=counties'
        state_table = self.table[self.table['state'] == state]
        if self.level == 'states':
            tab = state_table
        else:
            tab = state_table[state_table['county'] == county]
        tab = tab.sort_values(by='date')
        date = tab['date'].unique()
        return date[0], date[-1]

    def get(self, start_date, end_date, state, county=None):

        assert self.level == 'states' or county is not None, 'select a county for level=counties'
        state_table = self.table[self.table['state'] == state]
        if self.level == 'states':
            tab = state_table
        else:
            tab = state_table[state_table['county'] == county]
        start = datetime.datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.datetime.strptime(end_date, '%Y-%m-%d')
        st = tab.withColumn("mask", (tab.date >= start) & (tab.date <= end))
        st = st.filter(st.mask == True)

        #cases = spark.sql("select cases from st")
        st.cache()
        cases_rows = st.select("cases").collect()
        deaths_rows = st.select("deaths").collect()
        cases = np.array([x[0] for x in cases_rows])
        deaths = np.array([x[0] for x in deaths_rows])

        #deaths = spark.sql("select deaths from st");
        #cases = st['mask','cases'].to_numpy()
        #deaths = st['mask','deaths'].to_numpy()
        st.unpersist()
        return cases,deaths
