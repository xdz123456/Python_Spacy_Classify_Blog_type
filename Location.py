import string
import nltk
from nltk.tokenize import word_tokenize
import math
import json
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, atan2
import matplotlib.pyplot as plt
import seaborn as sns

# Download the stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Add some additional stop word
stop_words = stopwords.words('english')
stop_words.extend(
    ['from', 'subject', 're', 'edu', 'use', 'not', 'would', 'say', 'could', '_', 'be', 'know', 'good', 'go', 'get',
     'do', 'done', 'try', 'many', 'some', 'nice', 'thank', 'think', 'see', 'rather', 'easy', 'easily', 'lot', 'lack',
     'make', 'want', 'seem', 'run', 'need', 'even', 'right', 'line', 'even', 'also', 'may', 'take', 'come'])


# Calculate the R value
def calculate_R(current_term, high_or_low_all_terms, bg_all_terms):
    tf_HQ = high_or_low_all_terms.count(current_term)
    tf_BG = bg_all_terms.count(current_term)
    if tf_BG > 0:
        R = ((tf_HQ / len(high_or_low_all_terms))) / ((tf_BG / len(bg_all_terms)))
    else:
        R = 0
    return R


def calculate_newsworthy(tweets_terms, thresholds_HQ, thresholds_LQ, high_text_all_terms, low_text_all_terms,
                         bg_all_terms):
    total_S_HQ = 0
    total_S_LQ = 0

    for current_term in tweets_terms:
        R_HQ = calculate_R(current_term, high_text_all_terms, bg_all_terms)
        if R_HQ < thresholds_HQ:
            S_HQ = 0
        else:
            S_HQ = R_HQ

        R_LQ = calculate_R(current_term, low_text_all_terms, bg_all_terms)
        if R_LQ < thresholds_LQ:
            S_LQ = 0
        else:
            S_LQ = R_LQ

        total_S_HQ += S_HQ
        total_S_LQ += S_LQ

    newsworthy = math.log(((1 + total_S_HQ) / (1 + total_S_LQ)), 2)
    return newsworthy


# Read the high quality tweets
high_tweets = []
for line in open('highFileFeb', 'r'):
    high_tweets.append(json.loads(line))
highFile = pd.json_normalize(high_tweets)

# Read the low quality tweets
low_tweets = []
for line in open('lowFileFeb', 'r'):
    low_tweets.append(json.loads(line))
lowFile = pd.json_normalize(low_tweets)

low_text_terms = lowFile["text"].tolist()
high_text_terms = highFile["text"].tolist()

low_text_all_terms = []
high_text_all_terms = []

for i in range(0, len(low_text_terms)):
    low_text_terms[i] = low_text_terms[i].translate(str.maketrans('', '', string.punctuation))
    low_text_terms[i] = low_text_terms[i].lower()
    current_token = word_tokenize(low_text_terms[i])

    # Remove the stop word
    current_token_without_stopwords = [word for word in current_token if not word in stop_words]
    low_text_terms[i] = current_token_without_stopwords
    low_text_all_terms = low_text_all_terms + current_token_without_stopwords

for i in range(0, len(high_text_terms)):
    high_text_terms[i] = high_text_terms[i].translate(str.maketrans('', '', string.punctuation))
    high_text_terms[i] = high_text_terms[i].lower()

    # Remove Stop Words
    current_token = word_tokenize(high_text_terms[i])
    current_token_without_stopwords = [word for word in current_token if not word in stop_words]

    high_text_terms[i] = current_token_without_stopwords
    high_text_all_terms = high_text_all_terms + high_text_terms[i]

bg_text_all_terms = high_text_all_terms + low_text_all_terms

R_HQ_thresholds = 1.26
R_LQ_thresholds = 1.39

# Read the tweets by json_normalize
tweets = []
for line in open('geoLondonJan', 'r'):
    tweets.append(json.loads(line))
geoLondonJan = pd.json_normalize(tweets)
geo_term_list = geoLondonJan["text"].tolist()

for i in range(0, len(geo_term_list)):
    geo_term_list[i] = geo_term_list[i].translate(str.maketrans('', '', string.punctuation))
    geo_term_list[i] = geo_term_list[i].lower()
    current_token = word_tokenize(geo_term_list[i])

    # Remove the stop word
    current_token_without_stopwords = [word for word in current_token if not word in stop_words]
    geo_term_list[i] = current_token_without_stopwords

Newsworthy_geo = []

for current_tweets in geo_term_list:
    Newsworthy_geo += [
        calculate_newsworthy(current_tweets, R_HQ_thresholds, R_LQ_thresholds, high_text_all_terms, low_text_all_terms,
                             bg_text_all_terms)]

geoLondonJan["Newsworthy"] = Newsworthy_geo
geoLondonJan_with_newsworthy = geoLondonJan[(geoLondonJan["Newsworthy"] > 0)]

# Draw the bar chart
fig = plt.figure(figsize=(12, 8))
plt.hist(Newsworthy_geo, 50, linewidth=0)
plt.title("Distribution of geoLondonJan newsworthy")
plt.xlabel("Newsworthy")
plt.ylabel("Number of tweets")
plt.show()

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

# Get the km of london’s width and length in girds
London_width = int(np.ceil(get_distance([51.261318, 0.28036], [51.261318, -0.563])))
London_length = int(np.ceil(get_distance([51.686031, -0.563], [51.261318, -0.563])))

geoList = geoLondonJan_with_newsworthy["coordinates.coordinates"].tolist()

# Set a girds map
my_hot_map = np.zeros([London_length, London_width])

# Add value to the girds map
for i in range(0, len(geoList)):
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