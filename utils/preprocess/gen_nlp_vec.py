import getpass
import re
import numpy as np
from hdfs import InsecureClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import col
from globals import *


# Get the username
username = getpass.getuser()

# Initialize the hdfs client
hdfs_client = InsecureClient('http://localhost:9870', user=username)

# Initialize the spark session
spark = SparkSession.builder.appName("NLP vector generator").getOrCreate()


# Function to convert the description to a vector
def return_desc2vec(desc, word2vec):
    parts = re.split(' - | |\.|\\\|/|;|,|&|!|\?|\(|\)|\[|\]|\{|\}', desc)
    parts = [p.lower() for p in parts]
    vec_list = []
    for p in parts:
        if len(p) == 0:
            continue
        if p in word2vec:
            vec_list.append(word2vec[p])
    return np.mean(vec_list, axis=0)


# Function to load the word vectors
def load_word_vec(g_data_path):
    word2vec = {}
    glove_data = spark.read.text(g_data_path)
    for row in glove_data.collect():
        line = row[0]
        parts = line.replace('\r', '').replace('\n', '').split(' ')
        word2vec[parts[0]] = [float(parts[i]) for i in range(1, len(parts))]

    return word2vec


# Function to load the valid geohashes
def load_vld_geohash(geo_data_path):
    valid_geohashes = set()
    geohash_to_poi_data = spark.read.csv(geo_data_path, header=True, inferSchema=True)
    for row in geohash_to_poi_data.collect():
        line = ','.join([str(value) for value in row])
        valid_geohashes.add(line.split(',')[0])

    return valid_geohashes


# Function to generate a list of word vectors for each valid georegion (i.e. has poi data) from its descriptions
def gen_geo_to_vec(start, finish, valid_geohashes, word2vec):
    # Convert the datetime object to a string in the format 'YYYYMMDD'
    start_str = start.strftime('%Y%m%d')
    finish_str = finish.strftime('%Y%m%d')

    geo_to_vec = {}
    for c in cities:
        # Load the traffic data for the city
        # Generate the geohash for each record according to the latitudes and longitudes
        # Filter out the df with invalid geohashes
        df_vld_gh = spark.read.csv(f"hdfs://localhost:9000/data/temp/T_{c}_{start_str}_{finish_str}.csv/*", header=True,
                                   inferSchema=True)\
            .withColumn('geohash', geohash_udf(col('LocationLat').cast('float'), col('LocationLng').cast('float')))\
            .filter(col('geohash').isin(valid_geohashes))

        # Iterate over each record to append the NLP vector generated from the description to the list of vectors
        # of the corresponding geohash
        for row in df_vld_gh.rdd.collect():
            geohash = row.geohash

            if geohash not in geo_to_vec:
                geo_to_vec[geohash] = []

            geo_to_vec[geohash].append(return_desc2vec(row.Description, word2vec))

    return geo_to_vec


# Function to calculate the average vector for each geohash and save the result
def save_geo_to_vec(r_path, geo_to_vec):
    # If the file exists, delete it before writing to it
    if hdfs_client.status(r_path, strict=False):
        hdfs_client.delete(r_path)

    with hdfs_client.write(r_path, encoding='utf-8') as writer:
        writer.write('Geohash,vec\n')
        for g in geo_to_vec:
            vec = list(np.mean(geo_to_vec[g], axis=0))
            vec_str = [str(vec[i]) for i in range(len(vec))]
            vec_str = ' '.join(vec_str)
            writer.write(g + ',' + vec_str + '\n')


if __name__ == "__main__":
    # A time interval of length 1 year, to be used to generate description to vector for each geographical
    # region (or geohash)
    start = datetime(2017, 5, 1)
    finish = datetime(2018, 5, 31)

    # Extract the traffic data for each city during the time interval
    extract_t_data_4city(spark, t_data_path, start, finish)

    # Load the word vectors
    glove_data_path = 'hdfs://localhost:9000/data/glove.6B.100d.txt'
    word2vec = load_word_vec(glove_data_path)

    # Load valid geohashes
    geo_data_path = 'hdfs://localhost:9000/data/geohash_to_poi_vec.csv'
    valid_geohashes = load_vld_geohash(geo_data_path)

    # Generate a list of word vectors for each valid georegion (i.e. has poi data) from its descriptions
    geo_to_vec = gen_geo_to_vec(start, finish, valid_geohashes, word2vec)

    # Calculate the average vector for each geohash and save the result
    r_path = '/data/temp/geohash_to_text_vec.csv'
    save_geo_to_vec(r_path, geo_to_vec)
