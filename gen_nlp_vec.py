import getpass
import re
import pygeohash as gh
import numpy as np
from hdfs import InsecureClient
from pyspark.sql import SparkSession
from datetime import datetime
from pyspark.sql.functions import to_timestamp
from pyspark.sql.functions import udf
from pyspark.sql.types import StringType
from pyspark.sql.functions import lit
from pyspark import StorageLevel


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
	
	# Meta data for the cities of interest
	cities = {'LosAngeles': [33.700615, 34.353627, -118.683511, -118.074559], 
		       'Houston': [29.497907,30.129003,-95.797178,-94.988191],
		       'Austin': [30.079327, 30.596764,-97.968881,-97.504838],
		       'Dallas': [32.559567,33.083278,-97.036586,-96.428928],
		       'Charlotte': [34.970168,35.423667,-81.060925,-80.622687],
		       'Atlanta': [33.612410,33.916999,-84.575600,-84.231911]}
		       
	time_zones = {'Houston':'US/Central', 'Charlotte':'US/Eastern', 'Dallas':'US/Central',
		          'Atlanta':'US/Eastern', 'Austin':'US/Central', 'LosAngeles':'US/Pacific'}
		          
	# Each georegion is of 5km*5km
	geohash_prec = 5

	# A time interval of length 1 year, to be used to generate description to vector for each geographical region (or geohash) 
	start = datetime(2017, 5, 1)
	finish = datetime(2018, 5, 31)
	
	# Wrapper function for the geohash function
	geohash_udf = udf(encode_geohash, StringType())
	
	# Read in the traffic data
	mq = spark.read.csv('hdfs://localhost:9000/data/TrafficEvents_Aug16_Dec20_Publish.csv', header=True, inferSchema=True)

	# Convert the time for later calculations
	mq = mq.withColumn('StartTime(UTC)', to_timestamp(mq['StartTime(UTC)']))
	mq = mq.withColumn('EndTime(UTC)', to_timestamp(mq['EndTime(UTC)']))

	# Save the records that meet the spatial and temporal criteria for each city in a separate file
	for c in cities:
		crds = cities[c]
		subset_all = mq.filter((mq['StartTime(UTC)'] >= start) & (mq['StartTime(UTC)'] < finish) & 
		                (mq['LocationLat']>crds[0]) & (mq['LocationLat']<crds[1]) & (mq['LocationLng']>crds[2]) & 
		                (mq['LocationLng']<crds[3])) 
		subset_all.write.csv('hdfs://localhost:9000/data/temp/MQ_{}_20170501_20180531.csv'.format(c), header=True, mode='overwrite')
	
	# Load the word vectors
	word2vec = {}
	with hdfs_client.read('/data/glove.6B.100d.txt', encoding='utf-8') as reader:
		for line in reader:
		    parts = line.replace('\r', '').replace('\n', '').split(' ')
		    v = [float(parts[i]) for i in range(1, len(parts))]
		    word2vec[parts[0]] = v
	
	# Load valid geohashes
	valid_geohashes = set() 
	with hdfs_client.read('/data/geohash_to_poi_vec.csv', encoding='utf-8') as reader:
		for line in reader:
		    if 'Geohash' in line: continue
		    valid_geohashes.add(line.split(',')[0])
	
	# Convert the descriptions to a list of word vectors for each georegion
	geo_to_vec = {}
	for c in cities:
		records = spark.read.csv('hdfs://localhost:9000/data/temp/MQ_{}_20170501_20180531.csv'.format(c), header=True, inferSchema=True)
		
		records_gh = records.withColumn('start_gh', geohash_udf(records['LocationLat'].cast('float'), records['LocationLng'].cast('float'), lit(geohash_prec)))
		
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
