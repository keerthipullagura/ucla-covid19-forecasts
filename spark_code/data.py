import datetime
import warnings
import numpy as np

from pyspark.sql import SparkSession
from pyspark.sql.types import DateType

#This class initializes the data in form of NYTimes class where data is pulled from NYTimes for the cases and deaths based on the level state/county
#spark = SparkSession.builder.master("local[1]").appName("covid19seir").getOrCreate()
#Creating a spark session
spark = SparkSession.builder.getOrCreate()

class Data(object):
    def date_range(self):
        warnings.warn('Data range method does not implement')
        raise NotImplementedError

    def get(self, start_date, end_date):
        warnings.warn('Data get method does not implement')
        raise NotImplementedError


class NYTimes(Data):
    # Used to read and initialize state and county level data.
    # Changed this to use pyspark df so its loaded only when needed.
    def __init__(self, level='states'):
        assert level == 'states' or level == 'counties', 'level must be [states|counties]'

        if level == 'states':
            self.table = spark.read.format("csv").load("fetchedData/us-states.csv", format='csv', header='true', inferSchema='true').drop('fips')
        elif level == 'counties':
            self.table = spark.read.format("csv").load("fetchedData/us-counties.csv", format='csv', header='true', inferSchema='true').drop('fips')
        self.table = self.table.withColumn("casted_date",self.table['date'].cast(DateType()))
        self.level = level

        self.state_list = np.array(self.table.select('state').distinct().collect())

    #Utility method which reads the data from pyspark df based
    # on the level and filters based on a default limit of 10 identified for cases for a
    # particular date range
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

        st.cache()
        cases_rows = st.select("cases").collect()
        deaths_rows = st.select("deaths").collect()
        cases = np.array([x[0] for x in cases_rows])
        deaths = np.array([x[0] for x in deaths_rows])
        #Reading the cases and deaths from pyspark df and returning as numpy ndarray's.
        st.unpersist()
        return cases,deaths
