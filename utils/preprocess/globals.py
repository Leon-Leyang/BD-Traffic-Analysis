from datetime import datetime


# Metadata for the cities of interest
cities = {'LosAngeles': [33.700615, 34.353627, -118.683511, -118.074559],
          'Houston': [29.497907, 30.129003, -95.797178, -94.988191],
          'Austin': [30.079327, 30.596764, -97.968881, -97.504838],
          'Dallas': [32.559567, 33.083278, -97.036586, -96.428928],
          'Charlotte': [34.970168, 35.423667, -81.060925, -80.622687],
          'Atlanta': [33.612410, 33.916999, -84.575600, -84.231911]}

time_zones = {'Houston': 'US/Central', 'Charlotte': 'US/Eastern', 'Dallas': 'US/Central',
              'Atlanta': 'US/Eastern', 'Austin': 'US/Central', 'LosAngeles': 'US/Pacific'}

# A time interval of length 1 year, to be used to generate description to vector for each geographical region (or geohash)
start = datetime(2017, 5, 1)
finish = datetime(2018, 5, 31)