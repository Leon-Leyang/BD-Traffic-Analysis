import getpass

import pandas as pd
import numpy as np

import random
import os
import sys
import psutil

import matplotlib
import matplotlib.pyplot as plt
import math
from multiprocessing import cpu_count,Pool
import multiprocessing

from IPython.core.display_functions import display
from hdfs import InsecureClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf, col, lead
from pyspark.sql.types import StringType, IntegerType
from pyspark.sql.window import Window
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler
import pickle

# Get the user name
user_name = getpass.getuser()

# Initialize the hdfs client
hdfs_client = InsecureClient('http://localhost:9870', user=user_name)

# Initialize the spark session
spark = SparkSession.builder.appName("Data Cleaning").getOrCreate()
spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# put csv to hdfs for local testing
# put city_geo2vec folder under data folder locally
os.system('hdfs dfs -put /data/geohash_to_poi_vec.csv' + 'hdfs://localhost:9870/data/geohash_to_poi_vec.csv')
os.system('hdfs dfs -put /data/geohash_to_text_vec.csv' + 'hdfs://localhost:9870/data/geohash_to_text_vec.csv')
os.system('hdfs dfs -put /data/cite_geo2vec/Atlanta_geo2vec_201861-2018831.csv' + 'hdfs://localhost:9870/data/cite_geo2vec/Atlanta_geo2vec_201861-2018831.csv')
os.system('hdfs dfs -put /data/cite_geo2vec/Austin_geo2vec_201861-2018831.csv' + 'hdfs://localhost:9870/data/cite_geo2vec/Austin_geo2vec_201861-2018831.csv')
os.system('hdfs dfs -put /data/cite_geo2vec/Charlotte_geo2vec_201861-2018831.csv' + 'hdfs://localhost:9870/data/cite_geo2vec/Charlotte_geo2vec_201861-2018831.csv')
os.system('hdfs dfs -put /data/cite_geo2vec/Dallas_geo2vec_201861-2018831.csv' + 'hdfs://localhost:9870/data/cite_geo2vec/Dallas_geo2vec_201861-2018831.csv')
os.system('hdfs dfs -put /data/cite_geo2vec/Houston_geo2vec_201861-2018831.csv' + 'hdfs://localhost:9870/data/cite_geo2vec/Houston_geo2vec_201861-2018831.csv')
os.system('hdfs dfs -put /data/cite_geo2vec/LosAngeles_geo2vec_201861-2018831.csv' + 'hdfs://localhost:9870/data/cite_geo2vec/LosAngeles_geo2vec_201861-2018831.csv')

# geohash_map = pd.read_csv("../data/geohash_to_poi_vec.csv")
geohash_map = spark.read.csv('hdfs://localhost:9870/data/geohash_to_poi_vec.csv', header=True, inferSchema=True)
geohash_vec = geohash_map[[u'Amenity', u'Bump', u'Crossing', u'Give_Way',
                           u'Junction', u'Noexit', u'Railway', u'Roundabout', u'Station', u'Stop',
                           u'Traffic_Calming', u'Traffic_Signal', u'Turning_Circle',
                           u'Turning_Loop']]

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(geohash_vec.loc[:, 'Amenity':])
scaled_values = scaler.transform(geohash_vec.loc[:, 'Amenity':])
geohash_vec.loc[:, 'Amenity':] = scaled_values

geohash_dict = {}
for index, row in geohash_map.iterrows():
    geohash_dict[row.Geohash] = np.array(geohash_vec.iloc[index])

# f = open("geo_vect_dict.pkl", "wb")
# pickle.dump(geohash_dict, f)
# f.close()

# Define the HDFS file path
hdfs_pkl_path = 'geo_vect_dict.pkl'

# Transform to byte data
bytes_data = pickle.dumps(geohash_dict)

# Save byte data to HDFS
with hdfs_client.write(hdfs_pkl_path, overwrite=True) as writer:
    writer.write(bytes_data)

geo_dict = dict(zip(geohash_map.Geohash.unique(), range(len(geohash_map.Geohash.unique()))))
# f = open("geo_dict.pkl", "wb")
# pickle.dump(geo_dict, f)
# f.close()
hdfs_pkl_path = 'geo_dict.pkl'
bytes_data = pickle.dumps(geohash_dict)
with hdfs_client.write(hdfs_pkl_path, overwrite=True) as writer:
    writer.write(bytes_data)

# NLP_map = pd.read_csv("../data/geohash_to_text_vec.csv") # ???
NLP_map = spark.read.csv('hdfs://localhost:9870/data/geohash_to_text_vec.csv', header=True, inferSchema=True)

NLP_dict={}
iterRows = geohash_map.iterrows()
for index, row in NLP_map.iterrows():   # change to NLP_map
    NLP_dict[row.Geohash] = np.array([float(x) for x in row.vec.split(' ')])

# f = open("NLP_vect_dict.pkl","wb")
# pickle.dump(NLP_dict,f)
# f.close()
hdfs_pkl_path = 'NLP_vect_dict.pkl'
bytes_data = pickle.dumps(NLP_dict)
with hdfs_client.write(hdfs_pkl_path, overwrite=True) as writer:
    writer.write(bytes_data)

def clean_data(filepath, storename):

    df = spark.read.csv('hdfs://localhost:9870/data/'+filepath, header=True, inferSchema=True)
    # df = pd.read_csv(filepath)
    display(df.head())

    list_ = df.columns
    print(list_)

    temp_df = df.select(u'TimeStep', u'T-Accident', u'Geohash', u'HOD', u'DOW', u'DayLight',
                   u'T-BrokenVehicle', u'T-Congestion', u'T-Construction', u'T-Event',
                   u'T-FlowIncident', u'T-Other', u'T-RoadBlocked', u'W-Humidity',
                   u'W-Precipitation', u'W-Pressure', u'W-Temperature', u'W-Visibility',
                   u'W-WindSpeed', u'W-Rain', u'W-Snow', u'W-Fog', u'W-Hail')
    temp_df = temp_df.select("*").toPandas()
    temp_df.to_hdf(storename + '.h5', key='set1')
    # Save h5 to hdfs
    os.system('hdfs dfs -put ' + storename + '.h5' + 'hdfs://localhost:9870/' + storename + '.h5')
    display(temp_df.head())

    # print("zero accident =", float(df[df['T-Accident'] == 0].shape[0]) / df.shape[0])

    # f = open("geo_dict.pkl", "rb")
    # geo_dict = pickle.load(f)
    # f.close()

    hdfs_pkl_path = 'geo_dict.pkl'
    with hdfs_client.read(hdfs_pkl_path, encoding=None, delimiter=None) as reader:
        geo_dict = pickle.load(reader)

    def fun_hash(geohash):
        return geo_dict[geohash]

    df = df.withColumn('geohash_code', udf(lambda x: fun_hash(x), StringType())(col('Geohash')))
    temp_df = df.select(u'TimeStep', u'T-Accident', u'Geohash', u'geohash_code', u'HOD', u'DOW', u'DayLight',
                  u'T-BrokenVehicle', u'T-Congestion', u'T-Construction', u'T-Event',
                  u'T-FlowIncident', u'T-Other', u'T-RoadBlocked', u'W-Humidity',
                  u'W-Precipitation', u'W-Pressure', u'W-Temperature', u'W-Visibility',
                  u'W-WindSpeed', u'W-Rain', u'W-Snow', u'W-Fog', u'W-Hail')
    temp_df = temp_df.select("*").toPandas()
    temp_df.to_hdf(storename + '.h5', key='set2')
    # Save h5 to hdfs
    os.system('hdfs dfs -put '+storename + '.h5'+'hdfs://localhost:9870/'+storename + '.h5')

    # df = pd.read_hdf(storename + '.h5', key='set2')
    df = spark.read.format("hdf5").option('hdf5.filepath', 'hdfs://localhost:9870/'+storename + '.h5').load()
    display(df.head())

    def week_day(DOW):
        if DOW < 5:
            return 1
        else:
            return 0

    def shift(group):
        df_list = []
        for idx, df in group:
            df['predicted_accident'] = df['T-Accident'].shift(-1)
            df.drop(df.tail(1).index, inplace=True)
            df_list.append(df)
        return pd.concat(df_list)

    def time_interval(HOD):
        if HOD >= 6 and HOD < 10:
            return 0
        if HOD >= 10 and HOD < 15:
            return 1
        if HOD >= 15 and HOD < 18:
            return 2;
        if HOD >= 18 and HOD < 22:
            return 3
        else:
            return 4;

    def make_binary(d):
        if d > 0:
            return 1
        else:
            return 0

    # df['DOW_cat'] = df.apply(lambda row: week_day(row['DOW']), axis=1)
    # df['HOD_cat'] = df.apply(lambda row: time_interval(row['HOD']), axis=1)
    # df['T-Accident'] = df.apply(lambda row: make_binary(row['T-Accident']), axis=1)
    df = df.withColumn('DOW_cat', udf(lambda x: week_day(x), IntegerType())(col('DOW')))
    df = df.withColumn('HOD_cat', udf(lambda x: time_interval(x), IntegerType())(col('HOD')))
    df = df.withColumn('T-Accident', udf(lambda x: make_binary(x), IntegerType())(col('T-Accident')))
    w = Window.partitionBy('Group')
    # group = df.groupby('Geohash')
    # df = shift(group)
    df = df.withColumn('predicted_accident', lead(col('T-Accident'), 1).over(w))
    # delete last line
    df = df.filter(col('predicted_accident').isNotNull())
    temp_df = df[
        [u'TimeStep', u'predicted_accident', u'Geohash', u'geohash_code', u'HOD_cat', u'DOW_cat', u'T-Accident',
         u'DayLight',
         u'T-BrokenVehicle', u'T-Congestion', u'T-Construction', u'T-Event',
         u'T-FlowIncident', u'T-Other', u'T-RoadBlocked', u'W-Humidity',
         u'W-Precipitation', u'W-Pressure', u'W-Temperature', u'W-Visibility',
         u'W-WindSpeed', u'W-Rain', u'W-Snow', u'W-Fog', u'W-Hail']]
    temp_df = temp_df.select("*").toPandas()
    temp_df.to_hdf(storename + '.h5', key='set3')
    # Save h5 to hdfs
    os.system('hdfs dfs -put ' + storename + '.h5' + 'hdfs://localhost:9870/' + storename + '.h5')

cities = ['Atlanta', 'Austin', 'Charlotte', 'Dallas', 'Houston', 'LosAngeles']

for city in cities:
    clean_data("cite_geo2vec/{}_geo2vec_201861-2018831.csv".format(city), city)
