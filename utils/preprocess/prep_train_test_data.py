import pyspark
from pyspark.sql import functions, SparkSession
# from pyspark.ml.feature import MinMaxScaler
import numpy as np
import pandas as pd
import random
import os
import sys
import psutil

import matplotlib
import matplotlib.pyplot as plt
import math
from multiprocessing import cpu_count, Pool
import multiprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
import pickle
import getpass
from hdfs import InsecureClient

spark = SparkSession.builder.appName("step 4").config("spark.some.config.options", "some-value").getOrCreate()

# Get the user name
user_name = getpass.getuser()
# Initialize the hdfs client
hdfs_client = InsecureClient('http://localhost:9870', user=user_name)
# Define the HDFS file path
hdfs_pkl_path = '/path/to/hdfs/your_file.pkl'

# Read the .pkl file from HDFS
with client.read(hdfs_pkl_path, encoding=None, delimiter=None) as reader:
    geohash_dict = pickle.load(reader)

with client.read(hdfs_pkl_path, encoding=None, delimiter=None) as reader:
    geo_dict = pickle.load(reader)

with client.read(hdfs_pkl_path, encoding=None, delimiter=None) as reader:
    NLP_dict = pickle.load(reader)

# # Loading some necessary files locally
# f = open("geo_vect_dict.pkl","rb")
# geohash_dict = pickle.load(f)
# f.close()

# f = open("geo_dict.pkl","rb")
# geo_dict = pickle.load(f)
# f.close()

# f = open("NLP_vect_dict.pkl","rb")
# NLP_dict = pickle.load(f)
# f.close()


# # Helper functions for the parallel computing 
"""
cores = cpu_count() #Number of CPU cores on your system
partitions = cores

class WithExtraArgs(object):
     def __init__(self, func, **args):
         self.func = func
         self.args = args
     def __call__(self, df):
         return self.func(df, **self.args)

def applyParallel(data, func,pool,partition, kwargs):
     data_split = [data[i:i + partition] for i in range(0, len(data), partition)]
     #data_split = np.array_split(data, min(partitions,data.shape[0]))
     data =pool.map(WithExtraArgs(func, **kwargs), data_split)
     #data = pd.concat(pool.map(WithExtraArgs(func, **kwargs), data_split))
     return data

def parallelize(data, func,pool,partition):
     data_split = [data[i:i + partition] for i in range(0, len(data), partition)]
     #data_split = np.array_split(data, partitions)
     data =pool.map(func, data_split)
     return data
"""


def onhot_enoceder(train):
    myEncoder = OneHotEncoder(sparse=False)
    myEncoder.fit(train['HOD_cat'].values.reshape(-1, 1))

    onehot_encode = pd.concat([train.reset_index().drop('HOD_cat', 1),
                               pd.DataFrame(myEncoder.transform(train['HOD_cat'].values.reshape(-1, 1)),
                                            columns=['HOD_en0', 'HOD_en1', 'HOD_en2', 'HOD_en3', 'HOD_en4'])],
                              axis=1).reindex()
    return onehot_encode.drop('index', 1)


def create_train_set_aug_geo(frame_list, geomap):
    """
    process_name = str(multiprocessing.current_process())
    id = int(process_name.split(',')[0].split('-')[1])
    print("process ",id," started")
    """
    X_train = []
    y_train = []
    print("process list with length of ", len(frame_list))
    for frame in frame_list:
        training_set = frame.values
        # make sure there is unique geohash per frame
        # print frame.Geohash.iloc[0]
        geo_vec = geomap[frame.Geohash.iloc[0]]
        geo_code = geo_dict[frame.Geohash.iloc[0]]
        try:
            NLP_code = NLP_dict[frame.Geohash.iloc[0]]
        except:
            NLP_code = np.zeros(100)
        for i in range(8, training_set.shape[0]):
            if training_set[i, 1] > 0:
                a = np.concatenate((training_set[i - 8:i, 4:].flatten(), geo_vec), axis=0)
                a = np.concatenate((a, NLP_code), axis=0)
                a = np.append(a, geo_code)
                X_train.append(a)
                y_train.append(1)  # training_set[i, 1])

            elif random.uniform(0, 1) > 0.98:  # negative sampling for non-accident cases 
                a = np.concatenate((training_set[i - 8:i, 4:].flatten(), geo_vec), axis=0)
                a = np.concatenate((a, NLP_code), axis=0)
                a = np.append(a, geo_code)
                X_train.append(a)
                y_train.append(0)  # training_set[i, 1])
    return X_train, y_train


def create_sequences(df, geohash_dict):
    frame_list = []
    for idx, frame in df.groupby(df.Geohash):
        frame_list.append(frame)

    # pool = Pool(cores)
    # partition = int(np.ceil(float(len(frame_list))/partitions))
    # train_set = applyParallel (frame_list,create_train_set_aug_geo,pool,partition,{'geomap':geohash_dict.copy()})

    X_train, y_train = create_train_set_aug_geo(frame_list, geohash_dict)
    # pool.close()
    # pool.join()
    # X_train = []
    # y_train = []
    # for set_ in train_set:
    #    X_train.extend(set_[0])
    #    y_train.extend(set_[1])

    # X_train, y_train = np.array(X_train), np.array(y_train)
    return X_train, y_train


def train_data(filename):
    # # Try to load the h5 file in Spark
    # spark.conf.set("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
    # df = spark.read.format("hdf5").option("hdf5.filepath", filename+".h5").load()
    # max_timestep=df.select(functions.max('TimeStep')).collect()[0]['max(TimeStep)']
    # train_sc=df.filter(df['TimeStep']<=max_timestep*5/6)
    # test_sc=df.filter(df['TimeStep']>max_timestep*5/6)
    # train=train_sc.toPandas()
    # test=test_sc.toPandas()

    df = pd.read_hdf('hdf:...' + filename + '.h5', key='set3')  # fill in the absolute path in the hdfs file system
    display(df.head())
    df_normalize = df.copy()

    train = df_normalize[df_normalize.TimeStep <= df_normalize.TimeStep.max() * 5 / 6]
    test = df_normalize[df_normalize.TimeStep > df_normalize.TimeStep.max() * 5 / 6]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train.loc[:, 'T-BrokenVehicle':])
    scaled_values = scaler.transform(train.loc[:, 'T-BrokenVehicle':])
    train.loc[:, 'T-BrokenVehicle':] = scaled_values
    scaled_values = scaler.transform(test.loc[:, 'T-BrokenVehicle':])
    test.loc[:, 'T-BrokenVehicle':] = scaled_values
    display(test.head())

    train = onhot_enoceder(train)
    test = onhot_enoceder(test)

    X_train, y_train = create_sequences(train, geohash_dict)
    X_test, y_test = create_sequences(test, geohash_dict)
    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # suppose that we have a directory named train_set; in that directory we create several files per city to ...
    # ... represent its train and test data
    np.save('train_set/X_train_' + filename, X_train)
    print(X_train.shape)
    np.save('train_set/y_train_' + filename, y_train)
    print(y_train.shape)
    np.save('train_set/X_test_' + filename, X_test)
    print(X_test.shape)
    np.save('train_set/y_test_' + filename, y_test)
    print(y_test.shape)


cities = ['Atlanta', 'Austin', 'Charlotte', 'Dallas', 'Houston', 'LosAngeles']

for city in cities:
    train_data(city)  # creating training and test data for each city
