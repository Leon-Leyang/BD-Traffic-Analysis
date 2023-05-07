import getpass
import pickle
import random
import io
import numpy as np
import pandas as pd
from hdfs import InsecureClient
from pyspark.sql import SparkSession
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from globals import *


# Get the username
username = getpass.getuser()

# Initialize the hdfs client
hdfs_client = InsecureClient('http://localhost:9870', user=username)

# Initialize the spark session
spark = SparkSession.builder.appName("Train/test data generator").getOrCreate()


# Function to encode HOD_cat to HOD_enx for one-hot encoding
def onehot_encoder(train):
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(train['HOD_cat'].values.reshape(-1, 1))

    onehot_encode = pd.concat([train.reset_index().drop('HOD_cat', 1),
                               pd.DataFrame(encoder.transform(train['HOD_cat'].values.reshape(-1, 1)),
                                            columns=['HOD_en0', 'HOD_en1', 'HOD_en2', 'HOD_en3', 'HOD_en4'])],
                              axis=1).reindex()
    return onehot_encode.drop('index', 1)


def create_sequences(df, geohash_dict):
    frame_list = []
    for idx, frame in df.groupby(df.Geohash):
        frame_list.append(frame)

    x_train, y_train = create_train_set_aug_geo(frame_list, geohash_dict)

    return x_train, y_train


def create_train_set_aug_geo(frame_list, geo2poi):
    with hdfs_client.read(f'/data/temp/NLP_vect_dict.pickle') as reader:
        NLP_dict = pickle.load(reader)

    with hdfs_client.read(f'/data/temp/geo2idx.pickle') as reader:
        geo2idx = pickle.load(reader)

    x_train = []
    y_train = []

    for frame in frame_list:
        training_set = frame.values
        geo_vec = geo2poi[frame.Geohash.iloc[0]]
        geo_code = geo2idx[frame.Geohash.iloc[0]]
        try:
            NLP_code = NLP_dict[frame.Geohash.iloc[0]]
        except:
            NLP_code = np.zeros(100)

        for i in range(8, training_set.shape[0]):
            if training_set[i, 1] > 0:
                a = np.concatenate((training_set[i - 8:i, 4:].flatten(), geo_vec), axis=0)
                a = np.concatenate((a, NLP_code), axis=0)
                a = np.append(a, geo_code)
                x_train.append(a)
                y_train.append(1)
            elif random.uniform(0, 1) > 0.98:
                a = np.concatenate((training_set[i - 8:i, 4:].flatten(), geo_vec), axis=0)
                a = np.concatenate((a, NLP_code), axis=0)
                a = np.append(a, geo_code)
                x_train.append(a)
                y_train.append(0)

    return x_train, y_train


# Function to create train and test sets for the given city
def prep_train_test_data(c):
    with hdfs_client.read(f'/data/temp/geo2poi.pickle') as reader:
        geo2poi = pickle.load(reader)

    df = spark.read.csv(f"hdfs://localhost:9000/data/temp/{c}_geo2vec_cleaned.csv/*", header=True, inferSchema=True)
    df = df.toPandas()

    df_normalize = df.copy()
    train = df_normalize[df_normalize.TimeStep <= df_normalize.TimeStep.max() * 5 / 6]
    test = df_normalize[df_normalize.TimeStep > df_normalize.TimeStep.max() * 5 / 6]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train.loc[:, 'T-BrokenVehicle':])
    scaled_values = scaler.transform(train.loc[:, 'T-BrokenVehicle':])
    train.loc[:, 'T-BrokenVehicle':] = scaled_values
    scaled_values = scaler.transform(test.loc[:, 'T-BrokenVehicle':])
    test.loc[:, 'T-BrokenVehicle':] = scaled_values

    train = onehot_encoder(train)
    test = onehot_encoder(test)

    x_train, y_train = create_sequences(train, geo2poi)
    x_test, y_test = create_sequences(test, geo2poi)

    # Save numpy arrays to HDFS
    for name, array in zip(['x_train', 'y_train', 'x_test', 'y_test'], [x_train, y_train, x_test, y_test]):
        with io.BytesIO() as buffer:
            np.save(buffer, array)
            buffer.seek(0)
            hdfs_client.write(f'/data/train_set/{name}_{c}.npy', buffer)


if __name__ == "__main__":
    for c in cities:
        prep_train_test_data(c)
