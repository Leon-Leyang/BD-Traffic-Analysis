#!/usr/bin/env python
# coding: utf-8

import numpy as np
from datetime import datetime, timedelta
import pytz
import pygeohash as gh
import time
import numpy as np
import pytz
import pickle
import glob
import json
import os
from datetime import datetime, timedelta
from pyspark.sql import SparkSession
from pyspark.sql.functions import to_timestamp, date_format


spark = SparkSession.builder.appName("Python Spark SQL basic example").config("spark.some.config.option", "some-value").config("spark.debug.maxToStringFields", "100").getOrCreate()

geohash_prec = 5

class weather:
    date = ''
    temp = 0.0
    windchill = 0.0
    humid = 0.0
    pressure= 0.0
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


# Get the current time before reading the file
start_time = time.time()

acc_df = spark.read.option("header", "true")    .option("inferSchema", "true")    .option("delimiter", ",")    .csv("TrafficEvents_Aug16_Dec20_Publish.csv")

acc_df.head()

# Get the current time after reading the file
end_time = time.time()

# Calculate the elapsed time in seconds
elapsed_time = end_time - start_time

# Print the elapsed time
print(f"Time used to read the file: {elapsed_time:.2f} seconds")


acc_df = acc_df.withColumn("StartTime(UTC)",
    date_format(
        to_timestamp(acc_df["StartTime(UTC)"], "yyyy-MM-dd'T'HH:mm:ss"), 
        "yyyy-MM-dd'T'HH:mm:ss"
    ).alias('timestamp_value'))

acc_df = acc_df.withColumn("EndTime(UTC)",
    date_format(
        to_timestamp(acc_df["EndTime(UTC)"], "yyyy-MM-dd'T'HH:mm:ss"), 
        "yyyy-MM-dd'T'HH:mm:ss"
    ).alias('timestamp_value'))


cities = {'LosAngeles': [33.700615, 34.353627, -118.683511, -118.074559], 
           'Houston': [29.497907,30.129003,-95.797178,-94.988191],
           'Austin': [30.079327, 30.596764,-97.968881,-97.504838],
           'Dallas': [32.559567,33.083278,-97.036586,-96.428928],
           'Charlotte': [34.970168,35.423667,-81.060925,-80.622687],
           'Atlanta': [33.612410,33.916999,-84.575600,-84.231911]}

time_zones = {'Houston':'US/Central', 'Charlotte':'US/Eastern', 'Dallas':'US/Central',
              'Atlanta':'US/Eastern', 'Austin':'US/Central', 'LosAngeles':'US/Pacific'}

# time interval to sample data for 
start = datetime(2018, 6, 1)
finish = datetime(2018, 9, 2)

begin = datetime.strptime('2018-06-01 00:00:00', '%Y-%m-%d %H:%M:%S')
end = datetime.strptime('2018-08-31 23:59:59', '%Y-%m-%d %H:%M:%S')


for c in cities:
    crds = cities[c]
    subset_all = acc_df.where((acc_df['StartTime(UTC)'] >= start) & (acc_df['StartTime(UTC)'] < end)                              & (acc_df['LocationLat'] > crds[0]) & (acc_df['LocationLat'] < crds[1])                              & (acc_df['LocationLng'] > crds[2]) & (acc_df['LocationLng'] < crds[3]))
    
    
    subset_accidents = acc_df.where((acc_df['Type']=='Accident') & (acc_df['StartTime(UTC)'] >= start)                                     & (acc_df['StartTime(UTC)'] < finish) & (acc_df['LocationLat']>crds[0])                                     & (acc_df['LocationLat']<crds[1]) & (acc_df['LocationLng']>crds[2])                                     & (acc_df['LocationLng']<crds[3]))
    
    print(subset_all)
    
    subset_all.coalesce(1).write.option("header", "true")              .option("mapreduce.fileoutputcommitter.marksuccessfuljobs","false")              .csv('temporary/MQ_{}_all_time.csv'.format(c))


path = 'temporary/' #create this temporary directory inside the '/data' folder
mq_city2incidents = {}
for c in cities:
    incidents = []
    z = time_zones[c]
    
    with open(path + 'MQ_{}_all_time.csv'.format(c), 'r', encoding='utf-8') as file:
        header = False
        for line in file:
            if not header:
                header = True
                continue
            parts = line.replace('\r', '').replace('\n', '').split(',')
            
            ds = datetime.strptime(parts[5].replace('T',' '), '%Y-%m-%d %H:%M:%S')
            ds = ds.replace(tzinfo=pytz.utc)
            ds = ds.astimezone(pytz.timezone(z))
            
            de = datetime.strptime(parts[6].replace('T',' '), '%Y-%m-%d %H:%M:%S')
            de = de.replace(tzinfo=pytz.utc)
            de = de.astimezone(pytz.timezone(z))
            
            v = [parts[0], parts[1], float(parts[8]), float(parts[9]), ds, de]            
            incidents.append(v)
            
    mq_city2incidents[c] = incidents
    print ('MQ', c, len(incidents))
    print(incidents[0])


zone_to_be = {}

for z in ['US/Eastern', 'US/Central', 'US/Mountain', 'US/Pacific']:
    t_begin = begin.replace(tzinfo=pytz.timezone(z)) # t_begin : 根据utc转换后的开始时间
    t_end   = end.replace(tzinfo=pytz.timezone(z)) # t_end: 根据utc转换后的结束时间
    zone_to_be[z] = [t_begin, t_end]

name_conversion = {'Broken-Vehicle':'BrokenVehicle', 'Flow-Incident': 'FlowIncident', 'Lane-Blocked':'RoadBlocked'}


def return_interval_index(time_stamp, start, end):
    if time_stamp < start or time_stamp>end: 
        return -1
    index = int(((time_stamp - start).days*24*60 + (time_stamp-start).seconds/60)/15)
    return index

diff = int(((end - begin).days*24*60 + (end-begin).seconds/60)/15) # total_minutes/15 ==> number of 15 minutes intervals
print(f'diff: {diff}')

path = 'temporary/'
city_to_geohashes = {}
for c in cities: city_to_geohashes[c] = {}

start_timestamp = time.time()
ccnntt = 0

geocode_to_airport = {}
aiport_to_timezone = {}

for c in cities:
    z = time_zones[c]
    
    # add map-quest data
    with open(path + 'MQ_{}_all_time.csv'.format(c), 'r', encoding='utf-8') as file:
        header = False
        for line in file:
            if not header:
                header = True
                continue
            parts = line.replace('\r', '').replace('\n', '').split(',')
            
            ds = datetime.strptime(parts[5].replace('T',' '), '%Y-%m-%d %H:%M:%S')
            ds = ds.replace(tzinfo=pytz.utc)
            ds = ds.astimezone(pytz.timezone(z))
            s_interval = return_interval_index(ds, zone_to_be[z][0], zone_to_be[z][1])
            if s_interval==-1: continue
                
            de = datetime.strptime(parts[6].replace('T',' '), '%Y-%m-%d %H:%M:%S')
            de = de.replace(tzinfo=pytz.utc)
            de = de.astimezone(pytz.timezone(z))
            e_interval = return_interval_index(de, zone_to_be[z][0], zone_to_be[z][1])
            if e_interval == -1: e_interval = diff-1    
            
            start_gh = gh.encode(float(parts[8]), float(parts[9]), precision=geohash_prec)
            intervals = []
            if start_gh not in city_to_geohashes[c]:
                for i in range(diff): 
                    intervals.append({'Construction':0, 'Congestion':0, 'Accident':0, 'FlowIncident':0, 'Event':0, 
                                      'BrokenVehicle':0, 'RoadBlocked':0, 'Other':0})
            else:
                intervals = city_to_geohashes[c][start_gh]
            
            if parts[1] in name_conversion:
                tp = name_conversion[parts[1]]
            else: 
                tp = parts[1].split('-')[0]
                
            for i in range(s_interval, e_interval+1):                
                v = intervals[i]
                if tp in v: v[tp] = v[tp] + 1
                else: v['Other'] = v['Other'] + 1
                intervals[i] = v
                
                if tp == 'Accident': break # unlike other types of traffic events, 
                
            city_to_geohashes[c][start_gh] = intervals
            
            ap = parts[11]
            if len(ap) > 3:
                if start_gh not in geocode_to_airport:
                    geocode_to_airport[start_gh] = set([ap])
                else:
                    st = geocode_to_airport[start_gh]
                    st.add(ap)
                    geocode_to_airport[start_gh] = st
                aiport_to_timezone[ap] = z
  
    
    print('Done with {} in {:.1f} sec! there are {} geohashes with data!'.format(c, 
                                time.time()-start_timestamp, len(city_to_geohashes[c])))
    start_timestamp = time.time()


# load and sort relevant weather data
airports_to_observations = {}
for g in geocode_to_airport:
    aps = geocode_to_airport[g]
    for a in aps:
        if a not in airports_to_observations:
            airports_to_observations[a] = []

print(f'{len(airports_to_observations)} airports to collect data for!')
            
w_path = 'Sample_Weather/' # this directory contains weather observation records for each airport
# a sample data file can be find in '/data' directory (Sample_Weather.tar.gz)
airport_to_data = {}
for ap in airports_to_observations:
    data = []
    z = aiport_to_timezone[ap]
    print(f'Airport {ap}')
    header = ''
    if not os.path.isfile(w_path + ap + '.csv'):
        print(f'no file for Airport {ap}')
        continue
    with open(w_path + ap + '.csv', 'r') as file:
        for line in file:
            if 'Airport' in line: 
                header = line.replace('\r','').replace('\n','').replace(',Hour','')
                continue
            parts = line.replace('\r', '').replace('\n', '').split(',')
            try:
                w = weather(parts[1] + ' ' + parts[2].split(' ')[0] + ':00 ' + parts[2].split(' ')[1], parts[3], parts[4], 
                           parts[5], parts[6], parts[7], parts[8], parts[9], parts[10], parts[11], parts[12], z)   
                data.append(w)
            except:
                continue
    data.sort(key=lambda x:x.date)
    airport_to_data[ap] = data
    
print(f'\nData for {len(airport_to_data)} airport stations is loaded!')


for c in city_to_geohashes:
    for g in city_to_geohashes[c]:
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
#             print g, close_g, min_dist
            geocode_to_airport[g] = geocode_to_airport[close_g]


city_to_geohashes_to_weather = {}

for c in city_to_geohashes:
    start = time.time()
    geo2weather = {}
    for g in city_to_geohashes[c]:
        w_data = []
        for i in range(len(city_to_geohashes[c][g])):
            w_data.append({'Temperature':[], 'Humidity':[], 'Pressure':[], 'Visibility':[], 'WindSpeed':[], 
                          'Precipitation':[], 'Condition':set(), 'Event':set()})
        # populate weather data
        aps = geocode_to_airport[g]
        for a in aps:
            z = aiport_to_timezone[a]
            if a not in airport_to_data:
                continue
            a_w_data = airport_to_data[a]
            prev = 0
            for a_w_d in a_w_data:
                idx = return_interval_index(a_w_d.date, zone_to_be[z][0], zone_to_be[z][1])
                if idx >-1:
                    for i in range(prev, min(idx+1, len(w_data))):
                        _w = w_data[i]
                        
                        _tmp = _w['Temperature']
                        if a_w_d.temp > -1000:
                            _tmp.append(a_w_d.temp)
                            _w['Temperature'] = _tmp
                        
                        _hmd = _w['Humidity']
                        if a_w_d.humid > -1000:
                            _hmd.append(a_w_d.humid)
                            _w['Humidity'] = _hmd
                        
                        _prs = _w['Pressure']
                        if a_w_d.pressure > -1000:
                            _prs.append(a_w_d.pressure)
                            _w['Pressure'] = _prs
                        
                        _vis = _w['Visibility']
                        if a_w_d.visib > -1000:
                            _vis.append(a_w_d.visib)
                            _w['Visibility'] = _vis
                            
                        _wspd = _w['WindSpeed']
                        if a_w_d.windspeed > -1000:
                            _wspd.append(a_w_d.windspeed)
                            _w['WindSpeed'] = _wspd
                            
                        _precip = _w['Precipitation']
                        if a_w_d.precipitation > -1000:
                            _precip.append(a_w_d.precipitation)
                            _w['Precipitation'] = _precip
                            
                        _cond = _w['Condition']
                        _cond.add(a_w_d.condition)
                        _w['Condition'] = _cond
                        
                        _evnt = _w['Event']
                        _evnt.add(a_w_d.events)
                        _w['Event'] = _evnt
                        
                        w_data[i] = _w
                        
                    prev = idx+1
                                                
            
        geo2weather[g] = w_data
    city_to_geohashes_to_weather[c] = geo2weather
    print('Done with {} in {:.1f} sec!'.format(c, time.time()-start))


class dayLight:
    sunrise = []
    sunset = []
    def __init__(self, sunrise, sunset):
        self.sunrise = sunrise
        self.sunset = sunset
        
def return_time(x):
    try:
        h = int(x.split(':')[0])
        m = int(x.split(':')[1].split(' ')[0])
        if 'pm' in x and h < 12: h = h + 12
        return [h,m]
    except: return [0,0]

    
def returnDayLight(city, state, dt):
    sc = city + '-' + state
    days = city_days_time[sc]
    d = str(dt.year) + '-' + str(dt.month) + '-' + str(dt.day)
    if d in days:
        r = days[d]
        if ((dt.hour>r.sunrise[0] and dt.hour<r.sunset[0]) or
            (dt.hour>=r.sunrise[0] and dt.minute>=r.sunrise[1] and dt.hour<r.sunset[0]) or
            (dt.hour>r.sunrise[0] and dt.hour<=r.sunset[0] and dt.minute<r.sunset[1]) or 
            (dt.hour>=r.sunrise[0] and dt.minute>=r.sunrise[1] and dt.hour<=r.sunset[0] and dt.minute<r.sunset[1])):
            return '1'
        else: return '0'


city_days_time = {}

days = {}
city = ''
with open('sample_daylight.csv', 'r') as file: # you find daylight data for the selected 6 cities in this file
    for ln in file.readlines():
        parts = ln.replace('\r','').replace('\n','').split(',')

        if parts[0] != city:
            if len(city) > 0: 
                if city in city_days_time:
                    _days = city_days_time[city]
                    for _d in _days: days[_d] = _days[_d]
                city_days_time[city] = days

            city = parts[0]
            days = {}

        sunrise = return_time(parts[2])
        sunset  = return_time(parts[3])
        dl = dayLight(sunrise, sunset)
        days[parts[1]] = dl

if city in city_days_time:
    _days = city_days_time[city]
    for _d in _days: days[_d] = _days[_d]
city_days_time[city] = days


print('Successfully loaded daylight data for {} cities!'.format(len(city_days_time)))


# pre-load daylight mapping for different cities
city_to_index_to_daylight = {}
states = {'Houston':'TX', 'Charlotte':'NC', 'Dallas':'TX', 'Atlanta':'GA', 'Austin':'TX', 'LosAngeles':'CA'}
for c in cities:
    d_begin = begin.replace(tzinfo=pytz.timezone(time_zones[c]))
    d_end   = end.replace(tzinfo=pytz.timezone(time_zones[c]))
    index_to_daylight = {}
    index = 0
    while(d_begin < d_end):
        dl = returnDayLight(c, states[c], d_begin)
        index_to_daylight[index] = dl
        index += 1
        d_begin += timedelta(seconds=15*60)
    city_to_index_to_daylight[c] = index_to_daylight


# map each time-step to hour of day and day of the week; this should be consistent across different time-zones!
timestep_to_dow_hod = {}
d_begin = begin.replace(tzinfo=pytz.utc)
d_end   = end.replace(tzinfo=pytz.utc)
index = 0

while(d_begin < d_end):
    dow = d_begin.weekday()
    hod = d_begin.hour    
    timestep_to_dow_hod[index] = [dow, hod]
    
    d_begin += timedelta(seconds=15*60)    
    index += 1


traffic_tags = ['Accident', 'BrokenVehicle', 'Congestion', 'Construction', 'Event', 'FlowIncident', 'Other', 'RoadBlocked']
weather_tags = ['Condition', 'Event', 'Humidity', 'Precipitation', 'Pressure', 'Temperature', 'Visibility', 'WindSpeed']
poi_tags = []
start = time.time()
condition_tags = set()

for c in city_to_geohashes:
    # creating vector for each reion (geohash) during a 15 minutes time interval. Such vector contains time, traffic, and weather attributes. 
    writer = open('{}_geo2vec_{}-{}.csv'.format(c, str(begin.year)+str(begin.month)+str(begin.day),
                                                        str(end.year)+str(end.month)+str(end.day)), 'w')
    writer.write('Geohash,TimeStep,DOW,HOD,DayLight,T-Accident,T-BrokenVehicle,T-Congestion,T-Construction,'        'T-Event,T-FlowIncident,T-Other,T-RoadBlocked,W-Humidity,W-Precipitation,W-Pressure,'        'W-Temperature,W-Visibility,W-WindSpeed,W-Rain,W-Snow,W-Fog,W-Hail\n')
    
    traffic = city_to_geohashes[c]
    weather = city_to_geohashes_to_weather[c]        
    for g in traffic:
        vectors = []
        for i in range(len(traffic[g])):
            v = []
            for t in traffic_tags: v.append(traffic[g][i][t])
            v_w = [0,0,0,0] # for rain, snow, fog, and hail
            for w in weather_tags:
                if w=='Condition' or w=='Event':      
                    _tgs = weather[g][i][w]
                    for _tg in _tgs: 
                        if 'rain' in _tg.lower() or 'drizzle' in _tg.lower() or 'thunderstorm' in _tg.lower(): v_w[0] = 1
                        elif 'snow' in _tg.lower(): v_w[1] = 1
                        elif 'fog' in _tg.lower() or 'haze' in _tg.lower() or 'mist' in _tg.lower() or 'smoke' in _tg.lower(): v_w[2] = 1
                        elif 'hail' in _tg.lower() or 'ice pellets' in _tg.lower(): v_w[3] = 1                            
                elif len(weather[g][i][w]) == 0: v.append(0)
                else: v.append(np.mean(weather[g][i][w]))
            for _v_w in v_w: v.append(_v_w)
            vectors.append(v)
        
        for i in range(len(vectors)):
            v = vectors[i]
            v = [str(v[j]) for j in range(len(v))]
            v = ','.join(v)
            writer.write(g + ',' + str(i) + ',' + str(timestep_to_dow_hod[i][0]) + ',' + str(timestep_to_dow_hod[i][1]) 
                         + ',' + city_to_index_to_daylight[c][i] + ',' + v + '\n')
            
    writer.close()
    print('Done with {} in {:.1f} sec! #vectors {}!'.format(c, time.time()-start, len(traffic)*len(vectors)))
    start = time.time()
