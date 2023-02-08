import json
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import seaborn as sns


# Get the distance in km between two lon and lat point
def get_distance(point1, point2):
    # Earth Radius
    R = 6370
    # Get each points’ radians
    lat1 = radians(point1[0])
    lon1 = radians(point1[1])
    lat2 = radians(point2[0])
    lon2 = radians(point2[1])

    d_lon = lon2 - lon1
    d_lat = lat2 - lat1

    a = sin(d_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(d_lon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    # Get the distance
    distance = R * c
    return distance

# Read the tweets by json_normalize
tweets = []
for line in open('geoLondonJan', 'r'):
    tweets.append(json.loads(line))
geoLondonJan = pd.json_normalize(tweets)
geoList = geoLondonJan["coordinates.coordinates"]

# Get the km of london’s width and length in girds
London_width = int(np.ceil(get_distance([51.261318, 0.28036], [51.261318, -0.563])))
London_length = int(np.ceil(get_distance([51.686031, -0.563], [51.261318, -0.563])))

# Set a girds map
my_hot_map = np.zeros([London_length, London_width])

# Add value to the girds map
for i in range(0, geoList.size):
    km_x = int(np.ceil(get_distance([geoList[i][1], geoList[i][0]], [51.261318, geoList[i][0]])))
    km_y = int(np.ceil(get_distance([geoList[i][1], geoList[i][0]], [geoList[i][1], -0.563])))
    my_hot_map[km_x, km_y] += 1

# Draw the girds map in heatmap by using sns
sns.set()
fig = plt.figure()
sns_plot = sns.heatmap(my_hot_map, vmin=0, vmax=10)
plt.show()

# Set a dataframe for analysis the value count
my_df = pd.DataFrame(my_hot_map.flatten())

# Get the value count
value_count = my_df.value_counts()
# Sort the index by order
value_count = value_count.sort_index()
# Print the data distribution
print(value_count)

# Set List for draw bar chart
index_list = value_count.index.tolist()

# set bins
bin = []
for index in index_list:
    bin += [index[0]]

# Set counts
count = value_count.values

# Calculate data on girds
on_girds = 0
for i in range(0, len(bin)):
    on_girds += bin[i] * count[i]
print("Tweete on girds are " + str(on_girds))

# Draw the bar chart
fig = plt.figure(figsize=(12, 8))
plt.bar(bin, count, linewidth=0, width=8)
plt.ylim(0, 50)
plt.title("Distribution of number - London")
plt.xlabel("Number of tweets")
plt.ylabel("Number of grids")
plt.show()

