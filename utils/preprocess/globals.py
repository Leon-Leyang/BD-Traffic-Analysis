from datetime import datetime
from pyspark.sql.functions import to_timestamp


# Each georegion is of 5km*5km
geohash_prec = 5

# Metadata for the cities of interest
cities = {'LosAngeles': [33.700615, 34.353627, -118.683511, -118.074559],
          'Houston': [29.497907, 30.129003, -95.797178, -94.988191],
          'Austin': [30.079327, 30.596764, -97.968881, -97.504838],
          'Dallas': [32.559567, 33.083278, -97.036586, -96.428928],
          'Charlotte': [34.970168, 35.423667, -81.060925, -80.622687],
          'Atlanta': [33.612410, 33.916999, -84.575600, -84.231911]}

time_zones = {'Houston': 'US/Central', 'Charlotte': 'US/Eastern', 'Dallas': 'US/Central',
              'Atlanta': 'US/Eastern', 'Austin': 'US/Central', 'LosAngeles': 'US/Pacific'}


# The time interval of interest
begin = datetime.strptime('2018-06-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end = datetime.strptime('2018-08-31 23:59:59', '%Y-%m-%d %H:%M:%S')


t_data_path = 'hdfs://localhost:9000/data/TrafficEvents_Aug16_Dec20_Publish.csv'


# Function to extract the traffic data for each city
def extract_t_data_4city(spark, t_data_path, start, finish):
    # Convert the datetime object to a string in the format 'YYYYMMDD'
    start_str = start.strftime('%Y%m%d')
    finish_str = finish.strftime('%Y%m%d')

    # Read in the traffic data
    df = spark.read.csv(t_data_path, header=True, inferSchema=True)

    # Convert the time for later calculations
    df = df.withColumn('StartTime(UTC)', to_timestamp(df['StartTime(UTC)']))
    df = df.withColumn('EndTime(UTC)', to_timestamp(df['EndTime(UTC)']))

    # Save the records that meet the spatial and temporal criteria for each city in a separate file
    for c in cities:
        crds = cities[c]
        subset_all = df.filter((df['StartTime(UTC)'] >= start) &
                               (df['StartTime(UTC)'] < finish) &
                               (df['LocationLat'] > crds[0]) &
                               (df['LocationLat'] < crds[1]) &
                               (df['LocationLng'] > crds[2]) &
                               (df['LocationLng'] < crds[3]))
        subset_all.write.csv(f'hdfs://localhost:9000/data/temp/T_{c}_{start_str}__{finish_str}.csv'.format(c),
                             header=True, mode='overwrite')
