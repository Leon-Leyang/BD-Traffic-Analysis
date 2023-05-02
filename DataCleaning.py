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

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from sklearn.preprocessing import MinMaxScaler
import pickle

from pyspark.sql.types import *
from pyspark.shell import spark

from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

import findspark
findspark.init("/opt/homebrew/Cellar/spark-3.3.2-bin-hadoop3") # 指明SPARK_HOME

from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("Python Spark SQL basic example")\
.config("spark.some.config.option", "some-value")\
.config("spark.debug.maxToStringFields", "100")\
.getOrCreate()

geohash_map = pd.read_csv("data/geohash_to_poi_vec.csv")
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

f = open("geo_vect_dict.pkl", "wb")
pickle.dump(geohash_dict, f)
f.close()

geo_dict = dict(zip(geohash_map.Geohash.unique(), range(len(geohash_map.Geohash.unique()))))
f = open("geo_dict.pkl", "wb")
pickle.dump(geo_dict, f)
f.close()

NLP_map = pd.read_csv("data/geohash_to_text_vec.csv")

NLP_dict={}
for index, row in geohash_map.iterrows():
    NLP_dict[row.Geohash] = np.array([float(x) for x in row.vec.split(' ')])

f = open("NLP_vect_dict.pkl","wb")
pickle.dump(NLP_dict,f)
f.close()

def clean_data(filepath, storename):
    # df = pd.read_csv(filepath)
    data_schema = StructType([
        StructField('Geohash', StringType(), False),
        StructField('vec', IntegerType(), True)
    ])
    # input data
    # df = spark.read.format('csv').load(name="data/geohash_to_poi_vec.csv", schema=data_schema)
    # Create a SparkSession
    conf = SparkConf().setAppName("Read HDF5 file using Spark")
    sc = SparkContext(conf=conf)
    spark = SparkSession(sc)

    # Load the Hadoop InputFormat for HDF5 files
    spark.conf.set("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
    df = spark.read.format("hdf5").option("hdf5.filepath", "data/geohash_to_poi_vec.csv").load()

    # list_ = df.columns
    # print(list_)

    temp_df = df[[u'TimeStep', u'T-Accident', u'Geohash', u'HOD', u'DOW', u'DayLight',
                  u'T-BrokenVehicle', u'T-Congestion', u'T-Construction', u'T-Event',
                  u'T-FlowIncident', u'T-Other', u'T-RoadBlocked', u'W-Humidity',
                  u'W-Precipitation', u'W-Pressure', u'W-Temperature', u'W-Visibility',
                  u'W-WindSpeed', u'W-Rain', u'W-Snow', u'W-Fog', u'W-Hail']]
    temp_df.to_hdf(storename + '.h5', key='set1')

    print("zero accident =", float(df[df['T-Accident'] == 0].shape[0]) / df.shape[0])

    f = open("geo_dict.pkl", "rb")
    geo_dict = pickle.load(f)
    f.close()

    def fun_hash(geohash):
        return geo_dict[geohash]

    df['geohash_code'] = df.apply(lambda row: fun_hash(row['Geohash']), axis=1)
    temp_df = df[[u'TimeStep', u'T-Accident', u'Geohash', u'geohash_code', u'HOD', u'DOW', u'DayLight',
                  u'T-BrokenVehicle', u'T-Congestion', u'T-Construction', u'T-Event',
                  u'T-FlowIncident', u'T-Other', u'T-RoadBlocked', u'W-Humidity',
                  u'W-Precipitation', u'W-Pressure', u'W-Temperature', u'W-Visibility',
                  u'W-WindSpeed', u'W-Rain', u'W-Snow', u'W-Fog', u'W-Hail']]
    temp_df.to_hdf(storename + '.h5', key='set2')

    df = pd.read_hdf(storename + '.h5', key='set2')
    # display(df.head())

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

    df['DOW_cat'] = df.apply(lambda row: week_day(row['DOW']), axis=1)
    df['HOD_cat'] = df.apply(lambda row: time_interval(row['HOD']), axis=1)
    df['T-Accident'] = df.apply(lambda row: make_binary(row['T-Accident']), axis=1)
    group = df.groupby('Geohash')
    df = shift(group)
    temp_df = df[
        [u'TimeStep', u'predicted_accident', u'Geohash', u'geohash_code', u'HOD_cat', u'DOW_cat', u'T-Accident',
         u'DayLight',
         u'T-BrokenVehicle', u'T-Congestion', u'T-Construction', u'T-Event',
         u'T-FlowIncident', u'T-Other', u'T-RoadBlocked', u'W-Humidity',
         u'W-Precipitation', u'W-Pressure', u'W-Temperature', u'W-Visibility',
         u'W-WindSpeed', u'W-Rain', u'W-Snow', u'W-Fog', u'W-Hail']]
    temp_df.to_hdf(storename + '.h5', key='set3')