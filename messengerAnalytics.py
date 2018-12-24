import json
import datetime
import os, errno
import pickle
import math
import random
import numpy as np
import pandas as pd
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale
from textblob import TextBlob

MY_NAME = 'Raymond Situ'
DIR = "./data/messages/"
MIN_CONVO_LENGTH = 7000
POLARITY_BIN_SIZE = 200
K_POLARITY_MESSAGES = 10
METRIC_LIST = ['avg_msg', 'sent_receive_ratio', 'responsiveness', 'sent_receive_word_ratio', 'overall', 'avg_polarity', 'avg_photo_share']
stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don\'t', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'lol', 'yeah', 'u', 'ok', 'okay', 'yea', 'im', 'like', 'get', 'it\'s', 'ohh', 'got', 'ur', 'i\'m', 'oh', 'good', 'hey', 'sure', 'that\'s', 'ya', 'haha', 'also', 'you\'re', 'he\'s', 'dont', 'didn\'t','one', 'well', 'still','use',':)',':(','yup','uh','i\'ll', 'much', 'really', 'tho', 'kk', 'yes', 'man', 'yo'}

dataset = dict()
metrics = dict()

for m in METRIC_LIST:
    metrics[m] = dict()

polarity = dict()
polarity_global_df = pd.DataFrame.from_dict({'conversation':[],'sender':[],'message':[], 'polarity':[]})
photo_count = dict()

# helper to get name of other conversation participant
def get_name(participants):
    names = [person['name'] for person in participants]
    names.remove(MY_NAME)
    return str(names[0])

# read data
def read_json(foldername):
    message_dir = '{}{}/message.json'.format(DIR, foldername)
    if not os.path.exists(message_dir):
        return ('',[])
    with open(message_dir) as infile:
        data = json.load(infile)
        if len(data['participants']) == 2 and len(data['messages']) >= MIN_CONVO_LENGTH:
            name = get_name(data['participants'])
            return (name,data['messages'])
        else:
            return ('',[])

# count number of photos shared in a conversation
def count_pictures():
    global photo_count
    for foldername in os.listdir(DIR):
        message_dir = '{}{}/message.json'.format(DIR, foldername)
        if not os.path.exists(message_dir):
            continue
        with open(message_dir) as infile:
            data = json.load(infile)
            if len(data['participants']) != 2 or len(data['messages']) < MIN_CONVO_LENGTH:
                continue
            name = get_name(data['participants'])
            photo_directory = '{}{}/photos'.format(DIR, foldername)
            if os.path.exists(photo_directory):
                photo_count[name] = len(os.listdir(photo_directory))

# read data and load into a dict
def load_dict():
    print 'loading data'
    global dataset
    if os.path.exists('./data.pickle'):
        with open('./data.pickle', 'rb') as infile:
            dataset = pickle.load(infile)
    else:
        for foldername in os.listdir(DIR):
            name,json_data = read_json(foldername)
            if len(json_data) > 0:
                print name, len(json_data)
                dataset[name] = json_data
        pickle_dataset()
    count_pictures()

# cache the dict
def pickle_dataset():
    print 'pickling data'
    with open('data.pickle', 'wb') as outfile:
        pickle.dump(dataset, outfile, protocol=pickle.HIGHEST_PROTOCOL)

# Avg number of messages over active friendship period
def get_avg_msg(messages):
    last_ts = datetime.datetime.fromtimestamp(messages[0]['timestamp_ms']/1000.0)
    first_ts = datetime.datetime.fromtimestamp(messages[-1]['timestamp_ms']/1000.0)
    days = (last_ts - first_ts).days
    return float(len(messages))/days

# Avg number of photos shared over active friendship period
def get_avg_photos(messages, name):
    last_ts = datetime.datetime.fromtimestamp(messages[0]['timestamp_ms']/1000.0)
    first_ts = datetime.datetime.fromtimestamp(messages[-1]['timestamp_ms']/1000.0)
    days = (last_ts - first_ts).days
    return float(photo_count[name])/days

# Ratio of messages sent by me to messages received
def get_sent_receive_ratio(messages):
    sent = 0
    receive = 0
    for msg in messages:
        if msg['sender_name'] == MY_NAME:
            sent +=1
        else:
            receive +=1
    return float(sent)/receive

# Ratio of words in messages sent by me to words in messages received
def get_sent_receive_word_ratio(messages):
    sent_words = 0
    receive_words = 0
    for msg in messages:
        if 'content' not in msg.keys():
            continue
        filtered_words = [word for word in msg['content'].split(' ') if word.lower() not in stopwords]
        number_of_words = len(filtered_words)
        if number_of_words == 0:
            continue
        if msg['sender_name'] == MY_NAME:
            sent_words += number_of_words
        else:
            receive_words += number_of_words
    return float(sent_words)/receive_words

# get the avg polarity of messages
def get_avg_polarity(messages, name):
    result = []
    df_dict = dict()
    df_dict['conversation'] = []
    df_dict['sender'] = []
    df_dict['message'] = []
    df_dict['polarity'] = []
    df_dict['msg_length'] = []
    for msg in reversed(messages):
        if 'content' not in msg.keys():
            continue
        message_content = msg['content']
        message_tb = TextBlob(message_content)
        polarity = message_tb.sentiment.polarity
        result.append(polarity)
        if len(message_tb.words) > 5:
            df_dict['conversation'].append(name)
            df_dict['sender'].append(msg['sender_name'])
            df_dict['message'].append(filter_characters(message_content))
            df_dict['polarity'].append(polarity)
            df_dict['msg_length'].append(len(message_content))
    add_to_global_polarity(df_dict)
    return np.mean(result)

# Responsiveness (higher number means quicker to respond)
def get_responsiveness(messages):
    ts_list = []
    diff_list = []
    last_msg_from = MY_NAME
    for msg in messages:
        if msg['sender_name'] != last_msg_from:
            ts_list.append(msg['timestamp_ms'])
            last_msg_from = msg['sender_name']
    ts_list = sorted(ts_list)
    for i in range(len(ts_list) - 1):
        diff_list.append(1.0/(math.log(ts_list[i+1] - ts_list[i] + 1) +1))
    return np.mean(diff_list)

# overall score
def get_overall_friendship_score(name):
    max_msg = max(metrics['avg_msg'].values())
    max_photo = max(metrics['avg_photo_share'].values())
    max_responsiveness = max(metrics['responsiveness'].values())
    avg_msg = metrics['avg_msg'][name]/max_msg
    avg_photo_share = metrics['avg_photo_share'][name]/max_photo
    sent_receive_ratio = metrics['sent_receive_ratio'][name]
    sent_receive_word_ratio = metrics['sent_receive_word_ratio'][name]
    responsiveness = metrics['responsiveness'][name]/max_responsiveness
    return ((avg_msg + avg_photo_share*0.2)*responsiveness)/((sent_receive_ratio*sent_receive_word_ratio)**0.5)

# adds top k negative and positive polarity messages for each sender to the global df
def add_to_global_polarity(df_dict):
    global polarity_global_df
    df = pd.DataFrame.from_dict(df_dict)
    senders = df['sender'].unique()
    for sender in senders:
        polarity_global_df = polarity_global_df.append(
            df[df['sender'] == sender].sort_values(by=['polarity', 'msg_length'], ascending=False)
            .drop(columns=['msg_length'])
            .head(K_POLARITY_MESSAGES), sort=True)
        polarity_global_df = polarity_global_df.append(
            df[df['sender'] == sender].sort_values(by=['polarity', 'msg_length'], ascending=True)
            .drop(columns=['msg_length'])
            .head(K_POLARITY_MESSAGES), sort=True)

# export df of top negative and positive polarity messages to csv
def save_global_polarity_df():
    polarity_global_df[['conversation','sender','polarity','message']].to_csv('max_min_polarity_messages.csv',index=False,encoding='utf-8')

# group the messages into bins and take polarity of each bin
def plot_bin_polarity(messages, name):
    global polarity
    ts_list = []
    msg_list = []
    for msg in reversed(messages):
        if 'content' not in msg.keys():
            continue
        ts_list.append(msg['timestamp_ms'])
        msg_list.append(msg['content'])
    ts_bin = np.array_split(ts_list, POLARITY_BIN_SIZE)
    msg_bins = np.array_split(msg_list, POLARITY_BIN_SIZE)
    list_of_date = []
    list_of_avgs = []
    for txt in msg_bins:
        cleaned_txt = [t.rstrip('.') for t in txt]
        message_tb = TextBlob('.'.join(cleaned_txt))
        list_of_avgs.append(message_tb.sentiment.polarity)
    for ts in ts_bin:
        midpoint = (ts[-1] + ts[0])/2
        list_of_date.append(datetime.datetime.fromtimestamp(midpoint/1000.0).date())
    polarity[name] = list_of_avgs
    plot_polarity_trend_helper((list_of_date, list_of_avgs), name)

# Helper function to plot polarity trend for an individual with time
def plot_polarity_trend_helper(datapoints, name):
    plt.figure(figsize=(15,5))
    plt.plot(range(len(datapoints[1])), datapoints[1], label=name)
    plt.title('message polarity trend {}'.format(name))
    plt.xticks(range(0,210,10),datapoints[0][::9][:21],rotation=90)
    plt.yticks(np.arange(-0.6,0.7,0.05))
    plt.tight_layout()
    plt.ylabel('polarity [-1,1]')
    try:
        os.makedirs('./polarity_trends')
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
    plt.savefig('./polarity_trends/polarity_trend_{}.png'.format(name), dpi=200, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()

# plots the polarity trend for given names on a single graph
def plot_polarity_trend_multi(list_of_names, label):
    plt.figure(figsize=(15,5))
    for name in list_of_names:
        plt.plot(range(200), polarity[name], label=name)
    plt.title('message polarity trend {}'.format(label))
    plt.legend()
    plt.ylabel('polarity [-1,1]')
    plt.tight_layout()
    plt.tick_params(axis='x',labelbottom=False)
    plt.savefig('polarity_trend_{}.png'.format(label), dpi=200, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()

# based on overall metric ranking, plot the polarity of the top 3 and bottom 3 
def plot_best_worst(k):
    friend_ranking = sorted(metrics['overall'], key=metrics['overall'].get, reverse=True)
    best = friend_ranking[:k]
    worst = friend_ranking[-k:]
    plot_polarity_trend_multi(best,'best')
    plot_polarity_trend_multi(worst,'worst')

# get a freq dictionary of words
def get_words_freq(messages, k):
    word_freq = dict()
    for msg in messages:
        if 'content' not in msg.keys():
            continue
        filtered_words = [word.lower() for word in msg['content'].split(' ') if word.lower() not in stopwords]
        for word in filtered_words:
            word = filter_characters(word)
            if len(word) < 2:
                continue
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
    return word_freq

# preprocess string
def filter_characters(s):
    s = filter(lambda c: ord(c) < 128, s).replace('\\','')
    return s.decode('unicode-escape').encode('latin1')

# get avg frequence over all people
def get_average_freq_by_word(row):
    freq_total = 0
    for name in dataset.keys():
        freq_total += row[name]
    return freq_total/len(dataset.keys())

# generate heatmap of top k words
def top_words_heatmap(k):
    word_freq_dict = dict()
    for name in dataset.keys():
        word_freq_dict[name] = get_words_freq(dataset[name], k)

    df_dict = dict()
    top_words_set = set()
    for name in word_freq_dict:
        df_dict[name] = []
        top_words = sorted(word_freq_dict[name], key=word_freq_dict[name].get, reverse=True)[:k+1]
        top_words_set = top_words_set.union(set(top_words))
    
    df_dict['word'] = list(top_words_set)

    for name in word_freq_dict:
        df_dict[name] = []
        for word in df_dict['word']:
            if word not in word_freq_dict[name]:
                word_freq_dict[name][word] = 0
            df_dict[name].append(word_freq_dict[name][word])
        df_dict[name] = scale(df_dict[name])
        df_dict[name] = map(lambda x: (x+0.1), df_dict[name])

    df = pd.DataFrame.from_dict(df_dict)

    df['avg_freq'] = df.apply(get_average_freq_by_word, axis=1)
    df = df.sort_values('avg_freq', ascending=False)
    df = df.drop(columns=['avg_freq'])
    cols = df.columns.tolist()
    random.seed(7)
    random.shuffle(cols)
    df = df[cols]
    df = df.set_index('word')
    heatmap(df,k)

# plot and save heatmap
def heatmap(df,k):
    plt.figure(figsize=(6,10))
    sns.heatmap(df, cmap="Blues", yticklabels=list(df.index.values), square=True, cbar=False)
    plt.title('Word frequency heatmap')
    plt.tight_layout()
    plt.savefig('word_freq_{}.png'.format(k),bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()

# calculate metrics
def analyze(dataset):
    global metrics
    print 'analzing data'
    for name in dataset:
        print 'analyzing', name
        metrics['avg_msg'][name] = get_avg_msg(dataset[name])
        metrics['sent_receive_ratio'][name] = get_sent_receive_ratio(dataset[name])
        metrics['sent_receive_word_ratio'][name] = get_sent_receive_word_ratio(dataset[name])
        metrics['responsiveness'][name] = get_responsiveness(dataset[name])
        metrics['avg_polarity'][name] = get_avg_polarity(dataset[name], name)
        metrics['avg_photo_share'][name] = get_avg_photos(dataset[name], name)

    plot_metric('avg_polarity', 'Avg polarity of messages', 'polarity [-1,1]')
    plot_metric('avg_msg', 'Avg number of messages per day during friendship', 'messages')
    plot_metric('avg_photo_share', 'Avg number of photos shared per day during friendship', 'photos')
    plot_metric('sent_receive_ratio', 'Ratio of messages sent by me to messages received\n', 'ratio')
    plot_metric('sent_receive_word_ratio', 'Ratio of non-stopwords sent by me to non-stopwords received\n', 'ratio')

    for name in metrics['avg_msg']:
        metrics['overall'][name] = get_overall_friendship_score(name)

    fix_scale('overall')
    fix_scale('responsiveness')
    plot_metric('responsiveness', 'Responsiveness (higher number means quick response)', 'responsiveness')
    plot_metric('overall', 'Overall friendship score', 'how much you mean to me according\n to math')
    plot_best_worst(3)

# change the scale to max out at 1
def fix_scale(metric_name):
    global metrics
    max_value = max(metrics[metric_name].values())
    for name in metrics[metric_name]:
        metrics[metric_name][name] = round(metrics[metric_name][name]/max_value,3)

# store metrics to csv
def store_metric_results_csv():
    print 'storing metrics to csv'
    with open('metrics_summary.csv','w') as outfile:
        outfile.write('name,avg_msg,avg_photo_share,sent_receive_ratio,sent_receive_word_ratio,responsiveness,avg_polarity,overall_score\n')
        for name in dataset:
            outfile.write('{},{},{},{},{},{},{},{}\n'.format(
                name,
                metrics['avg_msg'][name],
                metrics['avg_photo_share'][name],
                metrics['sent_receive_ratio'][name],
                metrics['sent_receive_word_ratio'][name],
                metrics['responsiveness'][name],
                metrics['avg_polarity'][name],
                metrics['overall'][name]))

# plot and save metrics
def plot_metric(metric_name, title, y_axis):
    metric_data = metrics[metric_name]
    sorted_names = sorted(metric_data, key=metric_data.get, reverse=True)
    sorted_values = []
    for name in sorted_names:
        sorted_values.append(metric_data[name])
    y_pos = range(len(sorted_names))
    plt.bar(y_pos, sorted_values, align='center', width=0.5)
    plt.xticks(y_pos, sorted_names)
    plt.title(title)
    plt.ylabel(y_axis)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('{}.png'.format(metric_name), dpi=200)
    plt.cla()
    plt.clf()
    plt.close()

def main():
    load_dict()
    analyze(dataset)
    store_metric_results_csv()
    save_global_polarity_df()
    top_words_heatmap(10)

if __name__ == "__main__":
    main()
