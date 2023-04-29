import numpy as np
import csv
import time

# translate datetime format into integer value
def transform_time(dt):
    timeArray = time.strptime(dt, "%Y-%m-%d %H:%M:%S")
    timeStamp = int(time.mktime(timeArray))
    return timeStamp

# count the number of rows in a file
def iter_count(file_name):
    from itertools import (takewhile, repeat)
    buffer = 1024 * 1024
    with open(file_name, 'r', encoding='utf-8') as f:
        buf_gen = takewhile(lambda x: x, (f.read(buffer) for _ in repeat(None)))
        return sum(buf.count('\n') for buf in buf_gen)

# encode the boolean value
def make_boolean_label(dt, size, col_range):
    for col_index in col_range:
        col_index = col_index - 3

        for dt_row_index in range(size):
            if dt[dt_row_index][col_index] == "False":
                dt[dt_row_index][col_index] = 0
            else:
                dt[dt_row_index][col_index] = 1

# encode the day or night value
def make_day_night_label(dt, size, col_range):
    for col_index in col_range:
        col_index = col_index - 3
        for dt_row_index in range(size):
            if dt[dt_row_index][col_index] == "Day":
                dt[dt_row_index][col_index] = 0
            else:
                dt[dt_row_index][col_index] = 1

# encode the weather value
def make_weather_label(dt, size, weather_col_index):
    weather_col_index = weather_col_index - 3
    # gather different descriptions of the same weather type
    Clear = {'Clear', 'Fair', 'N/A Precipitation'}
    Cloudy = {'Overcast', 'Mostly Cloudy', 'Cloudy', 'Scattered Clouds', 'Partly Cloudy', 'Cloudy / Windy',
              'Partly Cloudy / Windy', 'Mostly Cloudy / Windy', 'Funnel Cloud'}

    Rain = {'Light Rain', 'Light Freezing Drizzle', 'Light Drizzle', 'Drizzle', 'Rain Showers',
            'Light Rain Showers', 'Light Freezing Rain', 'Rain', 'Heavy Rain', 'Heavy Drizzle',
            'Rain / Windy', 'Light Rain / Windy', 'Freezing Rain / Windy', 'Heavy Rain / Windy',
            'Showers in the Vicinity', 'Rain Shower', 'Light Rain Shower', 'Drizzle / Windy',
            'Light Drizzle / Windy', 'Squalls / Windy', 'Light Freezing Rain / Windy', 'Heavy Rain Shower / Windy',
            'Light Rain Shower / Windy', 'Freezing Rain', 'Freezing Drizzle', 'Heavy Freezing Drizzle',
            'Heavy Freezing Rain', 'Heavy Rain Shower', 'Heavy Rain Showers'}

    Dust = {'Haze', 'Smoke', 'Sand', 'Widespread Dust', 'Blowing Dust / Windy', 'Blowing Dust', 'Volcanic Ash',
            'Dust Whirls', 'Haze / Windy', 'Sand / Dust Whirlwinds', 'Smoke / Windy', 'Widespread Dust / Windy',
            'Sand / Windy', 'Duststorm', 'Sand / Dust Whirlwinds / Windy', 'Blowing Sand', 'Sand / Dust Whirls Nearby',
            'Light Haze'}

    Fog = {'Fog', 'Patches of Fog', 'Mist', 'Light Freezing Fog', 'Light Fog', 'Shallow Fog', 'Fog / Windy',
           'Partial Fog', 'Mist / Windy', 'Patches of Fog / Windy'}

    Hail = {'Light Ice Pellets', 'Ice Pellets', 'Small Hail', 'Heavy Ice Pellets', 'Hail'}

    Thunderstorm = {'Thunderstorms and Rain', 'Light Thunderstorms and Rain', 'Heavy Thunderstorms and Rain',
                    'Thunderstorm', 'Thunder', 'Thunder in the Vicinity', 'Light Rain with Thunder',
                    'Heavy Thunderstorms and Snow', 'Light Thunderstorms and Snow', 'Thunder / Windy',
                    'T-Storm / Windy', 'Heavy T-Storm', 'Heavy T-Storm / Windy', 'T-Storm', 'Light Snow with Thunder',
                    'Snow and Thunder / Windy', 'Thunder / Wintry Mix', 'Thunder / Wintry Mix / Windy',
                    'Heavy Thunderstorms with Small Hail'}

    Mix = {'Fair / Windy', 'Thunder and Hail', 'Thunder and Hail / Windy', 'Drizzle and Fog'}

    Snow = {'Snow', 'Light Snow', 'Blowing Snow', 'Heavy Snow', 'Snow Grains', 'Squalls', 'Light Snow Showers',
            'Wintry Mix', 'Wintry Mix / Windy', 'Light Snow / Windy', 'Heavy Snow / Windy', 'Snow / Windy',
            'Light Snow and Sleet', 'Blowing Snow / Windy', 'Snow and Sleet', 'Light Sleet', 'Light Snow Shower',
            'Light Sleet / Windy', 'Blowing Snow Nearby', 'Light Snow and Sleet / Windy', 'Heavy Sleet', 'Sleet',
            'Snow and Sleet / Windy', 'Sleet / Windy', 'Drifting Snow', 'Heavy Blowing Snow', 'Low Drifting Snow',
            'Light Blowing Snow', 'Heavy Snow with Thunder'}
    Tornado = {'Tornado'}

    # encode different weather types
    for dt_row_index in range(size):
        weather = dt[dt_row_index][weather_col_index]
        if weather in Clear:
            dt[dt_row_index][weather_col_index] = "1"
        elif weather in Cloudy:
            dt[dt_row_index][weather_col_index] = "2"
        elif weather in Rain:
            dt[dt_row_index][weather_col_index] = "3"
        elif weather in Dust:
            dt[dt_row_index][weather_col_index] = "4"
        elif weather in Fog:
            dt[dt_row_index][weather_col_index] = "5"
        elif weather in Hail:
            dt[dt_row_index][weather_col_index] = "6"
        elif weather in Thunderstorm:
            dt[dt_row_index][weather_col_index] = "7"
        elif weather in Mix:
            dt[dt_row_index][weather_col_index] = "8"
        elif weather in Snow:
            dt[dt_row_index][weather_col_index] = "9"
        elif weather in Tornado:
            dt[dt_row_index][weather_col_index] = "10"
        else:
            dt[dt_row_index][weather_col_index] = "11"

# encode the normal string value
def make_label(dt, size, col_range, dictionary_list):
    # adjust column index
    for col_index in col_range:
        if 22 > col_index > 10:
            col_index = col_index - 1
        elif 28 > col_index > 22:
            col_index = col_index - 2
        elif size > col_index > 28:
            col_index = col_index - 3

        index = 0
        # if dictionary is not empty, initialize the index of this key
        if dictionary_list[col_index]:
            index = max(dictionary_list[col_index].values())

        for dt_row_index in range(size):
            if dt[dt_row_index][col_index] in dictionary_list[col_index].keys():
                a = 1   # do nothing
            else:       # if new, not in dictionary
                index = index + 1
                dictionary_list[col_index][dt[dt_row_index][col_index]] = index
            dt[dt_row_index][col_index] = dictionary_list[col_index][dt[dt_row_index][col_index]]

    return dictionary_list

# separate the large original dataset into small sub-datasets
def cutDataset():
    FILE_SIZE = 100000      # each sub-dataset has FILE_SIZE rows
    row_iter = 0
    print_iter = 0
    features = []
    with open('../../../COMP4107 BD/CW/src/dataset/US_Accidents_Dec21_updated.csv', 'r', encoding='utf-8') as df:
        data = []
        totalRow = iter_count('../../../COMP4107 BD/CW/src/dataset/US_Accidents_Dec21_updated.csv')  # 2845343 rows in original large dataset
        # NUM_ROW_READ = 200000
        NUM_ROW_READ = totalRow         # decide how many rows to read
        for a in df:
            line = a.strip().split(',')
            # extract the first line, features
            if row_iter == 0:
                features = line
                row_iter = row_iter + 1
                continue
            data.append(line)
            # show the progress of reading files
            if row_iter % 5000 == 0:
                print(str(row_iter) + " / " + str(NUM_ROW_READ))
            # read first n rows
            if row_iter == NUM_ROW_READ:
                break
            row_iter = row_iter + 1

        # separate the whole dataset into small sub-datasets
        num_files = NUM_ROW_READ // FILE_SIZE
        for file_iter in range(num_files):
            fileName = "temp/subDataset_" + str(file_iter) + ".csv"
            with open(fileName, "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row_index in range(print_iter, print_iter + FILE_SIZE):
                    writer.writerow(data[row_index])

                print_iter = print_iter + FILE_SIZE

        # count the number of rest rows
        rest_row = NUM_ROW_READ - FILE_SIZE * num_files

        # deal with the rest rows
        if rest_row > 0:
            fileName = "temp/subDataset_" + str(num_files) + ".csv"
            with open(fileName, "a", newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                for row_index in range(print_iter, print_iter + rest_row):
                    if row_index>len(data)-1:
                        break
                    writer.writerow(data[row_index])
            num_files = num_files + 1
    print("Finish cutting dataset")
    return features, FILE_SIZE, num_files

# preprocess each sub-dataset and gather all output in one file
# share the same dictionary list
def PreProcess(file_index, dictionary_list):
    # read each sub-dataset and do preprocessing
    openFileName = "temp/subDataset_" + str(file_index) + ".csv"
    print("*** Open " + openFileName)
    with open(openFileName, 'r', encoding='utf-8') as df:
        LEN_FEATURE = 47
        totalRow = iter_count(openFileName)
        print("Total row = " + str(totalRow))
        data = []
        nulls = np.zeros([LEN_FEATURE])

        row_size = 0
        # Set number of rows to read
        NUM_ROW_READ = totalRow
        # NUM_ROW_READ = totalRow
        print("Rows to read = " + str(NUM_ROW_READ))
        for a in df:
            line = a.strip().split(',')
            data.append(line)
            for i in range(LEN_FEATURE):
                if len(line[i]) == 0:
                    nulls[i] = nulls[i] + 1
            row_size = row_size + 1
            if row_size % 5000 == 0:
                print(str(row_size) + " / " + str(NUM_ROW_READ))

        ###### delete Number, Wind_chill and Precipitation
        with open("../../../COMP4107 BD/CW/src/no_null_result.csv", "a", newline='', encoding='utf-8') as f:
            noNullResult = []
            writer = csv.writer(f)
            # manually delete certain columns, the automatic way is more complex and unnecessary
            # delete columns = 10(number), 22(wind chill), 28(Precipitation)
            raw_data = np.delete(data, [10, 22, 28], axis=1)
            # remove rows containing null values
            for row_index in range(NUM_ROW_READ):
                flag = False
                for column_index in range(len(nulls) - 3):
                    if len(raw_data[row_index][column_index]) == 0:
                        flag = True
                        break
                if flag == False:
                    noNullResult.append(raw_data[row_index])
                    #writer.writerow(raw_data[row_index])
        num_row = len(noNullResult)
        num_feature = len(noNullResult[0])
        print("row of no null result = " + str(num_row))
        print("column of no null result = " + str(num_feature))

        ######  translate String into Int (county, city to be continue)
        #   start_time = (0,2)
        #   end_time = (0,3)
        #   county = (0,13)

        # datetime translation & county labelling
        ORIGIN_DATETIME = transform_time('2016-02-01 00:00:00')
        for row_index in range(num_row):
            noNullResult[row_index][2] = str(
                transform_time(noNullResult[row_index][2].split(".")[0]) - ORIGIN_DATETIME)
            noNullResult[row_index][3] = str(
                transform_time(noNullResult[row_index][3].split(".")[0]) - ORIGIN_DATETIME)
            noNullResult[row_index][19] = str(
                transform_time(noNullResult[row_index][19].split(".")[0]) - ORIGIN_DATETIME)

        ## label side(12),county(14), state(15), Timezone(18),wind_direction(26)
        # encode string into integer value
        dictionary_list = make_label(noNullResult, num_row, [12, 14, 15, 18, 26], dictionary_list)

        # encode boolean
        make_boolean_label(noNullResult, num_row, range(30,43))

        # encode day and night label
        make_day_night_label(noNullResult, num_row, range(42,47))

        # encode weather label
        make_weather_label(noNullResult, num_row, 29)

        # remove description(9), street(10), city(12), zipcode(15), country(16), airport code(18)
        noNullResult = np.delete(noNullResult, [0, 9, 10, 12, 15, 16, 18], axis=1)

        # change end_time into time_duration (0,2)
        # change end_lng and end_lag into relative displacement (0,5)(0,6)
        for row_index in range(num_row):  # skip first row
            noNullResult[row_index][2] = str(int(noNullResult[row_index][2]) - int(noNullResult[row_index][1]))
            noNullResult[row_index][5] = str(float(noNullResult[row_index][5]) - float(noNullResult[row_index][3]))
            noNullResult[row_index][6] = str(float(noNullResult[row_index][6]) - float(noNullResult[row_index][4]))

        # write preprocessed data into one file
        with open("../../../COMP4107 BD/CW/src/transform_result_final.csv", "a", newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for line in noNullResult:
                writer.writerow(line)

        print("*** Finish Preprocess File No."+str(file_index))
        return dictionary_list
####    end of all functions


####    main program
#   Separate the original large dataset into small scaled sub-dataset
features, FILE_SIZE, num_files = cutDataset()

#   create public dictionary list
dictionary_list = []
for i in range(len(features)):
    dictionary_list.append({})

#   delete some features
features = np.delete(features, [10, 22, 28], axis=0)
features = np.delete(features, [0, 9, 10, 12, 15, 16, 18], axis=0)

#   end of time is changed to Duration
#   end of Lat is changed to Lat_Difference
#   end of Lng is changed to Lng_Difference
features[2] = "Duration"
features[5] = "Lat_Diff"
features[6] = "Lng_Diff"
features[12] = "Weather_Timestamp_Diff"

#   preprocess all sub-dataset and put all output into one file
for file_index in range(num_files):
    dictionary_list = PreProcess(file_index, dictionary_list)

#   change weather timestamp into difference value with start time
openFileName = "transform_result_final.csv"
data = []
with open(openFileName, 'r', encoding='utf-8') as df:
    LEN_FEATURE = 47
    totalRow = iter_count(openFileName)
    print("Total row = " + str(totalRow))

    # Set number of rows to read
    NUM_ROW_READ = totalRow
    # NUM_ROW_READ = totalRow
    for a in df:
        line = a.strip().split(',')
        line[12] = str(int(line[12]) - int(line[1]))
        data.append(line)

# write final csv file
print_iter = 0
PRINT_INTERVAL = 100000
print_iter_times = NUM_ROW_READ // PRINT_INTERVAL
for i in range(print_iter_times):
    with open("final_edited.csv", "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row_index in range(print_iter, print_iter + PRINT_INTERVAL):
            writer.writerow(data[row_index])

        print_iter = print_iter + PRINT_INTERVAL

rest_row = NUM_ROW_READ - PRINT_INTERVAL * print_iter_times
if rest_row > 0:
    with open("final_edited.csv", "a", newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        for row_index in range(print_iter, print_iter + rest_row):
            writer.writerow(data[row_index])

#   print final features of output
print(features)