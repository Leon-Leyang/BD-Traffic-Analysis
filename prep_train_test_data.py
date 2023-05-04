import pyspark
import numpy as np
import pandas as pd
import pickle
import getpass
import h5py
import random
import os
import sys
import psutil
import matplotlib
import matplotlib.pyplot as plt
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from pyspark.sql import functions, SparkSession
from hdfs import InsecureClient


spark = SparkSession.builder.appName("step 4").config("spark.some.config.options","some-value").getOrCreate()

# Get the user name
user_name = getpass.getuser()

# Initialize the hdfs client
hdfs_client = InsecureClient('http://localhost:9870', user=user_name)

# Define the HDFS file path
hdfs_pkl_path = '/path/to/hdfs/your_file.pkl'

# Read the .pkl file from HDFS
with hdfs_client.read(hdfs_pkl_path, encoding=None, delimiter=None) as reader:
    geohash_dict = pickle.load(reader)
    
# # Loading some necessary files 
# f = open("geo_vect_dict.pkl","rb")
# geohash_dict = pickle.load(f)
# f.close()

# f = open("geo_dict.pkl","rb")
# geo_dict = pickle.load(f)
# f.close()

# f = open("NLP_vect_dict.pkl","rb")
# NLP_dict = pickle.load(f)
# f.close()

df = pd.read_hdf('.../Atlanta.h5',key='set3') # the .h5 file contains raw traffic, weather, time, and POI data 
dislay(df.head())

spark.conf.set("spark.sql.sources.partitionColumnTypeInference.enabled", "false")
df = spark.read.format("hdf5").option("hdf5.filepath", ".../Atlanta.h5").load()  # change to absolute location
df.show()
