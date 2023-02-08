import json
import pandas as pd
import string
import matplotlib.pyplot as plt
import nltk
from nltk.tokenize import word_tokenize
import random
import math
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
    R = (tf_HQ / len(high_or_low_all_terms)) / (tf_BG / len(bg_all_terms))

    return R


# For normalize two list by randomly delete item in long list, until their len equal
def normal_term_list(input_list_1, input_list_2):
    list_1 = input_list_1.copy()
    list_2 = input_list_2.copy()
    if len(list_1) > len(list_2):
        delete_times = len(list_1) - len(list_2)
        for i in range(0, delete_times):
            index = random.randrange(len(list_1))
            list_1.pop(index)
    elif len(list_2) > len(list_1):
        delete_times = len(list_2) - len(list_1)
        for i in range(0, delete_times):
            index = random.randrange(len(list_2))
            list_2.pop(index)

    return list_1, list_2


# Draw the curve of Percentage Passing Threshold with different Threshold
def exp_best_thresholds(step, list_for_max, list_for_min, name):
    # init thresholds
    thresholds = 1.0
    thresholds_list = []
    num_max_list = []
    num_min_list = []
    num_max = 0
    num_min = 0
    diff = []

    while thresholds < max(list_for_min):
        # find last item which bigger than thresholds in list1_with_reverse_oreder
        for i in range(0, len(list_for_max)):
            if list_for_max[i] < thresholds:
                num_max_list += [i / len(list_for_max) * 100]
                num_max = i / len(list_for_max) * 100
                break
        # find last item which bigger than thresholds in list2_with_reverse_oreder
        for i in range(0, len(list_for_min)):
            if list_for_min[i] < thresholds:
                num_min_list += [i / len(list_for_min) * 100]
                num_min = i / len(list_for_min) * 100
                break
        thresholds_list += [thresholds]
        diff += [num_max - num_min]
        thresholds += step

    # Draw the plot figure
    plt.figure(figsize=(10, 10))
    plt.xlabel("Thresholds Value")
    plt.ylabel("Percentage greater than threshold")
    plt.title(name)
    plt.plot(thresholds_list, num_max_list, c='r')
    plt.plot(thresholds_list, num_min_list, c='g')
    plt.plot(thresholds_list, diff, c='b')
    plt.show()


def calculate_newsworthy(tweets_terms, thresholds_HQ, thresholds_LQ, high_text_all_terms, low_text_all_terms, bg_all_terms):
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

    newsworthy = math.log(((1 + total_S_HQ)/(1 + total_S_LQ)), 2)
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

R_HQ_high_terms = []
for term in high_text_all_terms:
    R_HQ_high_terms += [calculate_R(term, high_text_all_terms, bg_text_all_terms)]

R_HQ_low_terms = []
for term in low_text_all_terms:
    R_HQ_low_terms += [calculate_R(term, high_text_all_terms, bg_text_all_terms)]

# Normalize the term list
R_HQ_high_terms_norm, R_HQ_low_terms_norm = normal_term_list(R_HQ_high_terms, R_HQ_low_terms)

# Use the reverse sort to find the thresholds
R_HQ_high_terms_norm.sort(reverse=True)
R_HQ_low_terms_norm.sort(reverse=True)

exp_best_thresholds(0.01, R_HQ_high_terms_norm, R_HQ_low_terms_norm, "Thresholds Experiment of R_HQ with step by 0.01")

R_LQ_high_terms = []
for term in high_text_all_terms:
    R_LQ_high_terms += [calculate_R(term, low_text_all_terms, bg_text_all_terms)]

R_LQ_low_terms = []
for term in low_text_all_terms:
    R_LQ_low_terms += [calculate_R(term, low_text_all_terms, bg_text_all_terms)]

# Normalize the term list
R_LQ_high_terms_norm, R_LQ_low_terms_norm = normal_term_list(R_LQ_high_terms, R_LQ_low_terms)

# Use the reverse sort to find the thresholds
R_LQ_high_terms_norm.sort(reverse=True)
R_LQ_low_terms_norm.sort(reverse=True)

exp_best_thresholds(0.01, R_LQ_low_terms_norm, R_LQ_high_terms_norm, "Thresholds Experiment of R_LQ with step by 0.01")

# Obtain by analysis the figure
R_HQ_thresholds = 1.26
R_LQ_thresholds = 1.39


Newsworthy_high_list = []
Newsworthy_low_list = []

for current_tweets in high_text_terms:
    Newsworthy_high_list += [calculate_newsworthy(current_tweets, R_HQ_thresholds, R_LQ_thresholds, high_text_all_terms, low_text_all_terms, bg_text_all_terms)]

for current_tweets in low_text_terms:
    Newsworthy_low_list += [calculate_newsworthy(current_tweets, R_HQ_thresholds, R_LQ_thresholds, high_text_all_terms, low_text_all_terms, bg_text_all_terms)]

# Draw the bar chart
fig = plt.figure(figsize=(12, 8))
plt.hist(Newsworthy_high_list, 50, linewidth=0)
plt.title("Distribution of high file's newsworthy")
plt.xlabel("Newsworthy")
plt.ylabel("Number of tweets")
plt.show()

fig = plt.figure(figsize=(12, 8))
plt.hist(Newsworthy_low_list, 50, linewidth=0)
plt.title("Distribution of low file's newsworthy")
plt.xlabel("Newsworthy")
plt.ylabel("Number of tweets")
plt.show()

print("thresholds of HQ: " + str(R_HQ_thresholds))
print("thresholds of LQ: " + str(R_LQ_thresholds))
