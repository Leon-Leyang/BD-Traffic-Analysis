import getpass
import pytz
import pickle
from hdfs import InsecureClient
from datetime import time, timedelta
from haversine import haversine
from pyspark.sql import SparkSession
from pyspark.sql.functions import lit, when, col
from pyspark.sql.types import IntegerType, TimestampType, ArrayType
from pyspark.sql.functions import from_utc_timestamp, explode, length, concat
from globals import *

# Get the username
username = getpass.getuser()

# Initialize the hdfs client
hdfs_client = InsecureClient('http://localhost:9870', user=username)

# Initialize the spark session
spark = SparkSession.builder.appName("NLP vector generator").getOrCreate()

# time interval to sample data for
start = datetime(2018, 6, 1)
finish = datetime(2018, 9, 2)

begin = datetime.strptime('2018-06-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end = datetime.strptime('2018-08-31 23:59:59', '%Y-%m-%d %H:%M:%S')

# A dictionary that stores the begin and end time of each time zone
zone_to_be = {}
for z in ['US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific']:
    t_begin = begin.replace(tzinfo=pytz.timezone(z))
    t_end = end.replace(tzinfo=pytz.timezone(z))
    zone_to_be[z] = [t_begin, t_end]

# Calculate the total number of intervals
total_interval = int(((end - begin).days * 24 * 60 + (end - begin).seconds / 60) / 15)


# Class to store weather data
class Weather:
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
        self.date = date.replace(tzinfo=pytz.timezone(zone))
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
class DayLight:
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
def return_don(city_days_time, city, state, dt):
    sc = city + '-' + state
    days = city_days_time[sc]
    d = dt.strftime('%Y-%-m-%-d')
    if d in days:
        r = days[d]
        sunrise = time(*r.sunrise)
        sunset = time(*r.sunset)
        if sunrise <= dt.time() <= sunset:
            return '1'
        else:
            return '0'
    else:
        print('Error: ' + sc + ' ' + d)
        return '0'

# Function to process the traffic data
def proc_traffic_data(start, finish, begin, end):
    # Convert the datetime object to a string in the format 'YYYYMMDD'
    start_str = start.strftime('%Y%m%d')
    finish_str = finish.strftime('%Y%m%d')

    city_to_geohash = {}
    geocode_to_airport = {}
    airport_to_timezone = {}

    # Define the event types
    event_types = ['Construction', 'Congestion', 'Accident', 'FlowIncident', 'Event', 'BrokenVehicle', 'RoadBlocked',
                   'Other']

    # Mapping from the column names in the raw data to the column names in the processed data
    name_conversion = {'Broken-Vehicle': 'BrokenVehicle', 'Flow-Incident': 'FlowIncident',
                       'Lane-Blocked': 'RoadBlocked'}

    # Wrapper function for the return_interval_index function
    return_interval_index_udf = udf(return_interval_index, IntegerType())

    # Function to convert the column names in the raw data to the column names in the processed data
    name_conversion_udf = udf(lambda t: name_conversion.get(t, t.split('-')[0]), StringType())

    # Calculate the interval range for each record using an array
    # If the event is an accident, the interval range is [start, start + 1), else it is [start, end + 1)
    interval_range_udf = udf(lambda event_type, start, end: [start] if event_type == "Accident"
    else list(range(start, end + 1)), ArrayType(IntegerType()))

    for c in cities:
        geohash_to_traffic = {}
        city_to_geohash[c] = []
        z = time_zones[c]
        df = spark.read.csv(f"hdfs://localhost:9000/data/temp/T_{c}_{start_str}_{finish_str}.csv/*", header=True,
                            inferSchema=True)

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

        # Calculate the start and end interval for each record
        df = df.withColumn(
            "StartInterval",
            return_interval_index_udf("StartTime(Local)", lit(zone_to_be[z][0]), lit(zone_to_be[z][1]))
        ).withColumn(
            "EndInterval",
            return_interval_index_udf("EndTime(Local)", lit(zone_to_be[z][0]).cast(TimestampType()),
                                      lit(zone_to_be[z][1]).cast(TimestampType()))
        )

        # Replace any -1 values in the "EndInterval" column with a default value of total_interval - 1
        # Filter the records with StartInterval not equal to -1
        df = df.withColumn(
            "EndInterval",
            when(df["EndInterval"] == -1, total_interval - 1).otherwise(df["EndInterval"])
        ).filter(df["StartInterval"] != -1)

        # Create a new column to store the geohash of the start location
        df = df.withColumn("Geohash", geohash_udf(col('LocationLat').cast('float'), col('LocationLng').cast('float')))

        # Create a new column to store the processed event type
        df = df.withColumn("EventType", name_conversion_udf("Type"))

        # Add a column for each event type and set the value to 1 if the EventType matches the column, otherwise 0
        for et in event_types:
            if et != "Other":
                df = df.withColumn(et, when(df["EventType"] == et, 1).otherwise(0))

        # Create the 'Other' column
        # If the EventType does not match any of the defined event types, set it to 1, otherwise 0
        not_matched_event_types = ~df["EventType"].isin(event_types)
        df = df.withColumn("Other", when(not_matched_event_types, 1).otherwise(0))

        # Calculate the interval range for each record
        df = df.withColumn("IntervalRange", interval_range_udf("EventType", "StartInterval", "EndInterval"))

        # Explode the dataframe by IntervalRange and groupBy Geohash and Interval
        df = df.selectExpr("Geohash", "IntervalRange", "AirportCode", *event_types).withColumn("Interval", explode(
            "IntervalRange")).drop("IntervalRange")
        df_grouped = df.groupBy("Geohash", "Interval").agg({et: "sum" for et in event_types})

        # Collect the grouped rows to make a list of dictionaries of event type sums ordered by the time interval index
        # for each geohash and interval
        grouped_rows = df_grouped.collect()
        for row in grouped_rows:
            geohash = row.Geohash
            interval = row.Interval
            if geohash not in city_to_geohash[c]:
                city_to_geohash[c].append(geohash)
                geohash_to_traffic[geohash] = [{} for _ in range(total_interval)]

            event_type_sums = {et: row[f"sum({et})"] for et in event_types}
            try:
                geohash_to_traffic[geohash][interval] = event_type_sums
            except IndexError:
                print(f"Error: Interval {interval} is out of range for geohash {geohash} in city {c}.")

        # Check if the file exists in HDFS
        # If it exists, delete it
        if hdfs_client.status(f"/data/temp/{c}_geo2traffic.pickle", strict=False):
            hdfs_client.delete(f"/data/temp/{c}_geo2traffic.pickle")

        # Save the geohash_to_traffic data to HDFS using hdfs_client
        with hdfs_client.write(f"/data/temp/{c}_geo2traffic.pickle") as writer:
            pickle.dump(geohash_to_traffic, writer)

        # Update geocode_to_airport and aiport_to_timezone dictionaries
        df_airports = df.filter(length(df["AirportCode"]) > 3).select("Geohash", "AirportCode").distinct()
        for row in df_airports.collect():
            geocode_to_airport.setdefault(row.Geohash, set()).add(row.AirportCode)
            airport_to_timezone[row.AirportCode] = z

    return city_to_geohash, geocode_to_airport, airport_to_timezone


# Function to process the weather data
def proc_weather_data(airport_to_timezone):
    airport_to_data = {}

    for ap in airport_to_timezone:
        z = airport_to_timezone[ap]

        # Check if the file exists in HDFS
        if not hdfs_client.status(f"/data/Sample_Weather/{ap}.csv", strict=False):
            print(f"No file for airport {ap}")
            continue

        # Read the csv file from HDFS
        df = spark.read.csv(f"hdfs://localhost:9000/data/Sample_Weather/{ap}.csv", header=True, inferSchema=True) \
            .withColumn("Time", concat(col("Date"), lit(" "), col("Hour")))

        # Get the data for the airport
        data = []
        for row in df.collect():
            try:
                time = datetime.strptime(row["Time"], '%Y-%m-%d %I:%M %p')
                weather = Weather(
                    time,
                    row["Temp"],
                    row["WindChill"],
                    row["Humd"],
                    row["Pressure"],
                    row["Visib"],
                    row["WindSpeed"],
                    row["WindDir"],
                    row["Precipitation"],
                    row["Events"],
                    row["Conditions"],
                    z
                )
                data.append(weather)
            except ValueError as e:
                print(f'{e} Airport: {ap}. Time: {row["Time"]}')
                continue

        data.sort(key=lambda x: x.date)
        airport_to_data[ap] = data

    return airport_to_data


# Function to complement the missing airport data
def complement_missing_ap(city_to_geohash, geocode_to_airport):
    for c in city_to_geohash:
        for g in city_to_geohash[c]:
            if g not in geocode_to_airport:
                gc = gh.decode_exactly(g)[0:2]
                min_dist = 1000000000
                close_g = ''
                for _g in geocode_to_airport:
                    _gc = gh.decode_exactly(_g)[0:2]
                    dst = haversine(gc, _gc, 'km')
                    if dst < min_dist:
                        min_dist = dst
                        close_g = _g
                geocode_to_airport[g] = geocode_to_airport[close_g]

    return geocode_to_airport


# Function to assign weather data to each geohash
def assign_weather_data(city_to_geohash, airport_to_data, airport_to_timezone, geocode_to_airport):
    # Generator function to create the data list as an iterator
    def generate_data_intervals(total_interval):
        for _ in range(total_interval):
            yield {'Temperature': [], 'Humidity': [], 'Pressure': [], 'Visibility': [], 'WindSpeed': [],
                   'Precipitation': [], 'Condition': set(), 'Event': set()}

    for c in city_to_geohash:
        geohash_to_weather = {}
        for g in city_to_geohash[c]:
            data = list(generate_data_intervals(total_interval))

            ap_list = geocode_to_airport[g]
            for a in ap_list:
                z = airport_to_timezone[a]
                if a not in airport_to_data:
                    continue
                ap_data_list = airport_to_data[a]
                update_interval = 0
                for ap_data in ap_data_list:
                    data_idx = return_interval_index(ap_data.date, zone_to_be[z][0], zone_to_be[z][1])

                    if data_idx > -1:
                        for i in range(update_interval, min(data_idx + 1, len(data))):
                            if ap_data.temp > -1000:
                                data[i]['Temperature'].append(ap_data.temp)

                            if ap_data.humid > -1000:
                                data[i]['Humidity'].append(ap_data.humid)

                            if ap_data.pressure > -1000:
                                data[i]['Pressure'].append(ap_data.pressure)

                            if ap_data.visib > -1000:
                                data[i]['Visibility'].append(ap_data.visib)

                            if ap_data.windspeed > -1000:
                                data[i]['WindSpeed'].append(ap_data.windspeed)

                            if ap_data.precipitation > -1000:
                                data[i]['Precipitation'].append(ap_data.precipitation)

                            if ap_data.condition != '':
                                data[i]['Condition'].add(ap_data.condition)

                            if ap_data.events != '':
                                data[i]['Event'].add(ap_data.events)

                        update_interval = data_idx + 1

            geohash_to_weather[g] = data

        # Check if the file exists in HDFS
        # If it exists, delete it
        if hdfs_client.status(f"/data/temp/{c}_geo2weather.pickle", strict=False):
            hdfs_client.delete(f"/data/temp/{c}_geo2weather.pickle")

        # Save the geohash_to_weather data to HDFS using hdfs_client
        with hdfs_client.write(f"/data/temp/{c}_geo2weather.pickle") as writer:
            pickle.dump(geohash_to_weather, writer)


# Function to process the daylight data
def proc_daylight_data(dl_path):
    df = spark.read.csv(dl_path, header=False)

    # Function to get the dictionary of the daylight data
    # The dictionary is in the form of {city: {day: daylight}}
    city_days_time = {}
    for row in df.collect():
        city = row[0]
        if city not in city_days_time:
            city_days_time[city] = {}

        days = {}
        sunrise = return_time(row[2])
        sunset = return_time(row[3])
        dl = DayLight(sunrise, sunset)
        days[row[1]] = dl

        city_days_time[city].update(days)

    return city_days_time


# Function to calculate day or night for each interval in each city
def label_don_4interval(city_days_time):
    city_to_interval_to_don = {}
    states = {'Houston': 'TX', 'Charlotte': 'NC', 'Dallas': 'TX', 'Atlanta': 'GA', 'Austin': 'TX', 'LosAngeles': 'CA'}

    for c in cities:
        d_begin = begin.replace(tzinfo=pytz.timezone(time_zones[c]))
        d_end = end.replace(tzinfo=pytz.timezone(time_zones[c]))

        interval_to_don = {}
        interval = 0
        while d_begin < d_end:
            dl = return_don(city_days_time, c, states[c], d_begin)
            interval_to_don[interval] = dl
            interval += 1
            d_begin += timedelta(seconds=15*60)

        city_to_interval_to_don[c] = interval_to_don

    return city_to_interval_to_don


# Function to calculate the weekday or weekend and hour of a day for each interval
def label_dow_hod_4interval():
    interval_to_dow_hod = {}
    d_begin = begin.replace(tzinfo=pytz.utc)
    d_end = end.replace(tzinfo=pytz.utc)
    interval = 0

    while d_begin < d_end:
        dow = d_begin.weekday()
        hod = d_begin.hour
        interval_to_dow_hod[interval] = [dow, hod]
        interval += 1
        d_begin += timedelta(seconds=15*60)

    return interval_to_dow_hod


if __name__ == '__main__':
    # Extract the traffic data for each city during the time interval
    extract_t_data_4city(spark, t_data_path, start, finish)

    # Process the traffic data
    city_to_geohash, geocode_to_airport, airport_to_timezone = proc_traffic_data(start, finish, begin, end)

    # Complement the missing airport data
    geocode_to_airport = complement_missing_ap(city_to_geohash, geocode_to_airport)

    # Process the weather data
    airport_to_data = proc_weather_data(airport_to_timezone)

    # Assign weather data to each geohash
    assign_weather_data(city_to_geohash, airport_to_data, airport_to_timezone, geocode_to_airport)

    # Process the daylight data
    dl_path = "hdfs://localhost:9000/data/sample_daylight.csv"
    city_days_time = proc_daylight_data(dl_path)

    # Label day or night for each interval in each city
    city_to_interval_to_don = label_don_4interval(city_days_time)

    # Label weekday or weekend and hour of a day for each interval
    interval_to_dow_hod = label_dow_hod_4interval()
