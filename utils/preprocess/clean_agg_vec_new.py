import getpass
import pickle
from hdfs import InsecureClient
from pyspark.sql import SparkSession, Window
from pyspark.sql.functions import udf, col, lag
from pyspark.ml import Pipeline
from pyspark.ml.feature import MinMaxScaler, VectorAssembler
from pyspark.sql.types import ArrayType, DoubleType, IntegerType, StringType
from globals import *


# Get the username
username = getpass.getuser()

# Initialize the hdfs client
hdfs_client = InsecureClient('http://localhost:9870', user=username)

# Initialize the spark session
spark = SparkSession.builder.appName("Vector cleaner and aggregator").getOrCreate()


# Function that returns the index of the geohash
def geo_2_idx(geohash_to_idx, geohash):
    return geohash_to_idx[geohash]


# Function that returns 1 if the day of the week is a weekday, 0 otherwise
def get_week_day(dow):
    if dow < 5:
        return 1
    else:
        return 0


# Function that returns the time interval of the day
def get_time_interval(hod):
    if 6 <= hod < 10:
        return 0
    if 10 <= hod < 15:
        return 1
    if 15 <= hod < 18:
        return 2
    if 18 <= hod < 22:
        return 3
    else:
        return 4


# Function that returns 1 if the value is greater than 0, 0 otherwise
def make_binary(d):
    if d > 0:
        return 1
    else:
        return 0


# Function to process the poi data
def proc_poi_data(poi_path):
    # Select only the columns containing the POI vectors
    poi_cols = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'Noexit', 'Railway', 'Roundabout', 'Station',
                'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Circle', 'Turning_Loop']

    # Read the CSV file into a PySpark dataframe and select the POI columns
    df = spark.read.csv(poi_path, header=True).select("Geohash", *(col(c).cast("integer") for c in poi_cols))

    # Scale the POI vectors using MinMaxScaler
    vec_assembler = VectorAssembler(inputCols=poi_cols, outputCol="features")
    scaler = MinMaxScaler(inputCol="features", outputCol="scaled_features", min=0, max=1)
    pipeline = Pipeline(stages=[vec_assembler, scaler])
    model = pipeline.fit(df)
    df = model.transform(df)

    # Convert the scaled POI vectors to NumPy arrays and store in a dictionary
    geohash_to_poi = {}
    to_array_udf = udf(lambda row: row.toArray().tolist(), ArrayType(DoubleType()))
    df = df.withColumn("scaled_array", to_array_udf("scaled_features"))
    for row in df.collect():
        geohash_to_poi[row["Geohash"]] = row["scaled_array"]

    # Map each unique geohash to an index in the dictionary
    geohash_to_idx = {}
    for i, geohash in enumerate(geohash_to_poi.keys()):
        geohash_to_idx[geohash] = i

    # Check if the file exists in HDFS
    # If it exists, delete it
    if hdfs_client.status(f"/data/temp/geo2poi.pickle", strict=False):
        hdfs_client.delete(f"/data/temp/geo2poi.pickle")

    # Save the geohash_to_poi data to HDFS using hdfs_client
    with hdfs_client.write(f"/data/temp/geo2poi.pickle") as writer:
        pickle.dump(geohash_to_poi, writer)

    # Check if the file exists in HDFS
    # If it exists, delete it
    if hdfs_client.status(f"/data/temp/geo2idx.pickle", strict=False):
        hdfs_client.delete(f"/data/temp/geo2idx.pickle")

    # Save the geohash_to_poi data to HDFS using hdfs_client
    with hdfs_client.write(f"/data/temp/geo2idx.pickle") as writer:
        pickle.dump(geohash_to_idx, writer)


# Function to convert the NLP dataframe to a dictionary and save to HDFS
def convert_nlp_df2dict(nlp_path):
    # Read the CSV file into a PySpark dataframe
    df = spark.read.csv(nlp_path, header=True)

    # Define a UDF to convert the 'vec' column to a NumPy array
    to_array_udf = udf(lambda vec: [float(x) for x in vec.split(' ')], ArrayType(DoubleType()))
    df = df.withColumn("vec_array", to_array_udf("vec"))

    # Convert the dataframe to a dictionary
    NLP_dict = {}
    for row in df.collect():
        NLP_dict[row["Geohash"]] = row["vec_array"]

    # Check if the file exists in HDFS
    # If it exists, delete it
    if hdfs_client.status(f"/data/temp/NLP_vect_dict.pickle", strict=False):
        hdfs_client.delete(f"/data/temp/NLP_vect_dict.pickle")

    # Save the NLP dictionary to HDFS using hdfs_client
    with hdfs_client.write(f"/data/temp/NLP_vect_dict.pickle") as writer:
        pickle.dump(NLP_dict, writer)


# Function to clean the data
def clean_data(c):
    df = spark.read.csv(f'hdfs://localhost:9000/data/temp/{c}_geo2vec.csv', header=True, inferSchema=True)

    with hdfs_client.read(f'/data/temp/geo2idx.pickle') as reader:
        geohash_to_idx = pickle.load(reader)

    geo_2_idx_udf = udf(lambda x: geo_2_idx(geohash_to_idx, x), StringType())
    get_week_day_udf = udf(get_week_day, IntegerType())
    get_time_interval_udf = udf(get_time_interval, IntegerType())
    make_binary_udf = udf(make_binary, IntegerType())

    df = df.withColumn("geohash_code", geo_2_idx_udf(df["Geohash"]))
    df = df.withColumn("DOW_cat", get_week_day_udf(df["DOW"]))
    df = df.withColumn("HOD_cat", get_time_interval_udf(df["HOD"]))
    df = df.withColumn("T-Accident", make_binary_udf(df["T-Accident"]))

    w = Window.partitionBy("Geohash").orderBy("TimeStep")
    df = df.withColumn("predicted_accident", lag("T-Accident", -1).over(w))
    df = df.dropna(subset=("predicted_accident",))

    # Select the desired columns
    selected_columns = [
        'TimeStep', 'predicted_accident', 'Geohash', 'geohash_code', 'HOD_cat', 'DOW_cat', 'T-Accident', 'DayLight',
        'T-BrokenVehicle', 'T-Congestion', 'T-Construction', 'T-Event', 'T-FlowIncident', 'T-Other', 'T-RoadBlocked',
        'W-Humidity', 'W-Precipitation', 'W-Pressure', 'W-Temperature', 'W-Visibility', 'W-WindSpeed', 'W-Rain',
        'W-Snow', 'W-Fog', 'W-Hail'
    ]
    df = df.select(selected_columns)

    df.write.csv(f'hdfs://localhost:9000/data/temp/{c}_geo2vec_cleaned.csv', header=True, mode="overwrite")


if __name__ == '__main__':
    # Process the POI data
    poi_path = "hdfs://localhost:9000/data/geohash_to_poi_vec.csv"
    proc_poi_data(poi_path)

    # Process the NLP data
    nlp_path = "hdfs://localhost:9000/data/temp/geohash_to_text_vec.csv"
    convert_nlp_df2dict(nlp_path)

    # Clean the data
    for c in cities:
        clean_data(c)
