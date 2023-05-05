import pytz
from datetime import datetime
from globals import *


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
