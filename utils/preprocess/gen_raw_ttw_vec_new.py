import getpass
import pytz
from hdfs import InsecureClient
from globals import *
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, from_utc_timestamp

# Get the username
username = getpass.getuser()

# Initialize the hdfs client
hdfs_client = InsecureClient('http://localhost:9870', user=username)

# Initialize the spark session
spark = SparkSession.builder.appName("NLP vector generator").getOrCreate()


# Class to store weather data
class weather:
    date = ''
    temp = 0.0
    windchill = 0.0
    humid = 0.0
    pressure = 0.0
    visib = 0.0
    windspeed = 0.0
    winddir = ''
    precipitation = 0.0
    events = ''
    condition = ''

    def __init__(self, date, temp, windchill, humid, pressure, visib, windspeed, winddir,
                 precipitation, events, condition, zone):
        self.date = datetime.strptime(date, '%Y-%m-%d %I:%M:%S %p')
        self.date = self.date.replace(tzinfo=pytz.timezone(zone))
        self.temp = float(temp)
        self.windchill = float(windchill)
        self.humid = float(humid)
        self.pressure = float(pressure)
        self.visib = float(visib)
        self.windspeed = float(windspeed)
        self.winddir = winddir
        self.precipitation = float(precipitation)
        self.events = events
        self.condition = condition


# Class to store daylight data
class dayLight:
    sunrise = []
    sunset = []

    def __init__(self, sunrise, sunset):
        self.sunrise = sunrise
        self.sunset = sunset


# Function to return the index of the interval that the time stamp falls into
def return_interval_index(time_stamp, start, end):
    if time_stamp < start or time_stamp > end:
        return -1
    index = int(((time_stamp - start).days * 24 * 60 + (time_stamp - start).seconds / 60) / 15)
    return index


# Function to return the time in 24h format
def return_time(x):
    try:
        h = int(x.split(':')[0])
        m = int(x.split(':')[1].split(' ')[0])
        if 'pm' in x and h < 12:
            h = h + 12
        return [h, m]
    except:
        return [0, 0]


# Function to return if the given time is day or night
def returnDayLight(city_days_time, city, state, dt):
    sc = city + '-' + state
    days = city_days_time[sc]
    d = str(dt.year) + '-' + str(dt.month) + '-' + str(dt.day)
    if d in days:
        r = days[d]
        if ((r.sunrise[0] < dt.hour < r.sunset[0]) or
                (r.sunrise[0] <= dt.hour < r.sunset[0] and dt.minute >= r.sunrise[1]) or
                (r.sunrise[0] < dt.hour <= r.sunset[0] and dt.minute < r.sunset[1]) or
                (r.sunrise[0] <= dt.hour <= r.sunset[0] and r.sunrise[1] <= dt.minute < r.sunset[1])):
            return '1'
        else:
            return '0'


# Function to return three dictionaries
def proc_traffic_data(start, finish, begin, end):
    # Convert the datetime object to a string in the format 'YYYYMMDD'
    start_str = start.strftime('%Y%m%d')
    finish_str = finish.strftime('%Y%m%d')

    city_to_geohashes = {}
    geocode_to_airport = {}
    airport_to_timezone = {}

    # Calculate the total number of intervals
    total_interval = int(((end - begin).days * 24 * 60 + (end - begin).seconds / 60) / 15)

    for c in cities:
        z = time_zones[c]
        df = spark.read.csv(f"hdfs://localhost:9000/data/temp/T_{c}_{start_str}_{finish_str}.csv/*",
                                      header=True, inferSchema=True)

        # Convert StartTime(UTC) to local time in the specified timezone
        df = df.withColumn(
            "StartTime(Local)",
            from_utc_timestamp(to_timestamp("StartTime(UTC)", "yyyy/MM/dd HH:mm:ss"), z)
        ).withColumn(
            "EndTime(Local)",
            from_utc_timestamp(to_timestamp("EndTime(UTC)", "yyyy/MM/dd HH:mm:ss"), z)
        )

        # Convert timestamp columns to Python datetime objects
        df = df.withColumn("StartTime(Local)", df["StartTime(Local)"].cast("timestamp")).withColumn(
            "EndTime(Local)", df["EndTime(Local)"].cast("timestamp")
        )



if __name__ == '__main__':
    # time interval to sample data for
    start = datetime(2018, 6, 1)
    finish = datetime(2018, 9, 2)

    begin = datetime.strptime('2018-06-01 00:00:00', '%Y-%m-%d %H:%M:%S')
    end = datetime.strptime('2018-08-31 23:59:59', '%Y-%m-%d %H:%M:%S')

    # Extract the traffic data for each city during the time interval
    extract_t_data_4city(spark, t_data_path, start, finish)

    # Process the traffic data
    proc_traffic_data(start, finish, begin, end)
