import getpass
import re
import pygeohash as gh
import numpy as np
from hdfs import InsecureClient
from pyspark.sql import SparkSession
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import lit
from globals import *


def encode_geohash(lat, lng, precision):
    return gh.encode(lat, lng, precision=precision)


def return_desc2vec(desc, word2vec):
    parts = re.split(' - | |\.|\\\|/|;|,|&|!|\?|\(|\)|\[|\]|\{|\}', desc)
    parts = [p.lower() for p in parts]
    v = []
    for p in parts:
        if len(p) == 0: continue
        if p in word2vec: v.append(word2vec[p])
    if len(v) == 0: print(desc)
    v = np.mean(v, axis=0)
    return v


def calc_avg_vec(geo_vecs):
    geo, vecs = geo_vecs
    avg_vec = np.mean(vecs, axis=0)
    return geo, avg_vec


if __name__ == "__main__":
    # Get the user name
    user_name = getpass.getuser()

    # Initialize the hdfs client
    hdfs_client = InsecureClient('http://localhost:9870', user=user_name)

    # Initialize the spark session
    spark = SparkSession.builder.appName("NLP vector generator").getOrCreate()

    # Wrapper function for the geohash function
    geohash_udf = udf(encode_geohash, StringType())

    # A time interval of length 1 year, to be used to generate description to vector for each geographical
    # region (or geohash)
    start = datetime(2017, 5, 1)
    finish = datetime(2018, 5, 31)

    # Extract the traffic data for each city during the time interval
    extract_t_data_4city(spark, t_data_path, start, finish)

    # Load the word vectors
    word2vec = {}
    glove_data = spark.read.text('hdfs://localhost:9000/data/glove.6B.100d.txt')
    for row in glove_data.collect():
        line = row[0]
        parts = line.replace('\r', '').replace('\n', '').split(' ')
        v = [float(parts[i]) for i in range(1, len(parts))]
        word2vec[parts[0]] = v

    # Load valid geohashes
    valid_geohashes = set()
    geohash_to_poi_data = spark.read.csv('hdfs://localhost:9000/data/geohash_to_poi_vec.csv', header=True, inferSchema=True)
    for row in geohash_to_poi_data.collect():
        line = ','.join([str(value) for value in row])
        if 'Geohash' in line:
            continue
        valid_geohashes.add(line.split(',')[0])

    # Convert the descriptions to a list of word vectors for each georegion
    geo_to_vec = {}
    for c in cities:
        records = spark.read.csv('hdfs://localhost:9000/data/temp/MQ_{}_20170501_20180531.csv'.format(c), header=True,
                                 inferSchema=True)

        records_gh = records.withColumn('start_gh', geohash_udf(records['LocationLat'].cast('float'),
                                                                records['LocationLng'].cast('float'),
                                                                lit(geohash_prec)))

        records_vld_gh = records_gh.filter(records_gh['start_gh'].isin(valid_geohashes))

        # Iterate over each record
        for row in records_vld_gh.rdd.collect():
            start_gh = row.start_gh
            if start_gh in geo_to_vec:
                mat = geo_to_vec[start_gh]
            else:
                mat = []

            # Append the NLP vector of the description to mat
            mat.append(return_desc2vec(row.Description, word2vec))

            # Update geo_to_vec with the new list of vectors for this geohash
            geo_to_vec[start_gh] = mat

    # Save the data
    with hdfs_client.write('/data/temp/geohash_to_text_vec.csv', encoding='utf-8') as writer:
        writer.write('Geohash,vec\n')
        for g in geo_to_vec:
            vec = list(np.mean(geo_to_vec[g], axis=0))
            v = [str(vec[i]) for i in range(len(vec))]
            v = ' '.join(v)
            writer.write(g + ',' + v + '\n')
