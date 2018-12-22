# -*- coding: utf-8 -*-
import json
import datetime
import os
import pickle
import math
import random
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import scale
from textblob import TextBlob

DIR = "./data/messages/"
MIN_CONVO_LENGTH = 7000
stopwords = {'ourselves', 'hers', 'between', 'yourself', 'but', 'again', 'there', 'about', 'once', 'during', 'out', 'very', 'having', 'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its', 'yours', 'such', 'into', 'of', 'most', 'itself', 'other', 'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him', 'each', 'the', 'themselves', 'until', 'below', 'are', 'we', 'these', 'your', 'his', 'through', 'don\'t', 'nor', 'me', 'were', 'her', 'more', 'himself', 'this', 'down', 'should', 'our', 'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had', 'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them', 'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does', 'yourselves', 'then', 'that', 'because', 'what', 'over', 'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you', 'herself', 'has', 'just', 'where', 'too', 'only', 'myself', 'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being', 'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it', 'how', 'further', 'was', 'here', 'than', 'lol', 'yeah', 'u', 'ok', 'okay', 'yea', 'im', 'like', 'get', 'it\'s', 'ohh', 'got', 'ur', 'i\'m', 'oh', 'good', 'hey', 'sure', 'that\'s', 'ya', 'haha', 'also', 'you\'re', 'he\'s', 'dont', 'didn\'t','one', 'well', 'still','use',':)',':(','yup','uh','i\'ll', 'much', 'really', 'tho', 'kk', 'yes', 'man', 'yo'}

dataset = dict()
metrics = dict()
metrics['avg_msg'] = dict()
metrics['sent_receive_ratio'] = dict()
metrics['responsiveness'] = dict()
metrics['sent_receive_word_ratio'] = dict()
metrics['overall'] = dict()
metrics['avg_polarity'] = dict()

polarity = dict()

# read data
def read_json(foldername):
    try:
        with open('{}{}/message.json'.format(DIR, foldername)) as infile:
            data = json.load(infile)
            if len(data['participants']) == 2 and len(data['messages']) > MIN_CONVO_LENGTH:
                return data['messages']
            else:
                return []
    except:
        return []

# read data and load into a dict
def load_dict():
    global dataset
    if os.path.exists('./data.pickle'):
        with open('./data.pickle', 'rb') as infile:
            dataset = pickle.load(infile)
    else:
        for foldername in os.listdir(DIR):
            json_data = read_json(foldername)
            if len(json_data) > 0:
                print foldername.split('_')[0], len(json_data)
                dataset[(foldername.split('_')[0]).lower()] = json_data

# cache the dict
def pickel_dataset():
    with open('data.pickle', 'wb') as outfile:
        pickle.dump(dataset, outfile, protocol=pickle.HIGHEST_PROTOCOL)

# Avg number of messages over active friendship period
def get_avg_msg(messages):
    last_ts = datetime.datetime.fromtimestamp(messages[0]['timestamp_ms']/1000.0)
    first_ts = datetime.datetime.fromtimestamp(messages[-1]['timestamp_ms']/1000.0)
    days = (last_ts - first_ts).days
    return round(float(len(messages))/days,2)

# Ratio of messages sent by me to messages received
def get_sent_receive_ratio(messages):
    sent = 0
    receive = 0
    for msg in messages:
        if msg['sender_name'] == 'Raymond Situ':
            sent +=1
        else:
            receive +=1
    return round(float(sent)/receive,2)

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
        if msg['sender_name'] == 'Raymond Situ':
            sent_words += number_of_words
        else:
            receive_words += number_of_words
    return round(float(sent_words)/receive_words,2)

# Responsiveness (higher number means quicker to respond)
def get_responsiveness(messages):
    ts_list = []
    diff_list = []
    last_msg_from = 'Raymond Situ'
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
    avg_msg = metrics['avg_msg'][name]
    sent_receive_ratio = metrics['sent_receive_ratio'][name]
    sent_receive_word_ratio = metrics['sent_receive_word_ratio'][name]
    responsiveness = metrics['responsiveness'][name]
    return (avg_msg*responsiveness)/((sent_receive_ratio*sent_receive_word_ratio)**0.5)

# get the avg polarity of messages
def get_avg_polarity(messages, name):
    result = list()
    for msg in reversed(messages):
        if 'content' not in msg.keys():
            continue
        message_text = TextBlob(msg['content'])
        polarity = message_text.sentiment.polarity
        result.append(polarity)
    bin_polarity(result, name)
    return round(np.mean(result),2)

# split the polarity of messages into bins and take avg of each bin
def bin_polarity(polarity_lst, name):
    global polarity
    bins = np.array_split(polarity_lst, 200)
    list_of_avgs = []
    for b in bins:
        list_of_avgs.append(np.mean(b))
    polarity[name] = list_of_avgs
    plot_polarity_trend(list_of_avgs, name.lower())

# plot individual polarity trend for an individual
def plot_polarity_trend(list_of_avgs, name):
    plt.figure(figsize=(15,5))
    plt.plot(range(200), list_of_avgs, label=name)
    plt.title('message polarity trend {}'.format(name))
    plt.tick_params(axis='x',labelbottom=False)
    plt.yticks(np.arange(min(list_of_avgs), max(list_of_avgs)+0.1, 0.025))
    plt.tight_layout()
    plt.savefig('./polarity_trends/polarity_trend_{}.png'.format(name), dpi=200, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()

# plots the polarity trend for given names on a single graph
def plot_polarity_trend_multi(list_of_names, label):
    plt.figure(figsize=(15,5))
    for name in list_of_names:
        plt.plot(range(200), polarity[name], label=name.lower())
    plt.title('message polarity trend {}'.format(label))
    plt.legend()
    plt.tight_layout()
    plt.tick_params(axis='x',labelbottom=False)
    plt.savefig('polarity_trend_{}.png'.format(label), dpi=200, bbox_inches='tight')
    plt.cla()
    plt.clf()
    plt.close()

# get a freq dictionary of words
def get_words_freq(messages, k):
    word_freq = dict()
    for msg in messages:
        if 'content' not in msg.keys():
            continue
        filtered_words = [word.lower() for word in msg['content'].split(' ') if word.lower() not in stopwords]
        for word in filtered_words:
            if len(word) < 2:
                continue
            if not filter_s(word):
                continue
            if word not in word_freq:
                word_freq[word] = 0
            word_freq[word] += 1
    return word_freq

# check if word contains non-ascii characters
def filter_s(word):
    return all(ord(c) < 128 for c in word)

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
    df.columns = [x.lower() for x in df.columns]
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
    for name in dataset:
        metrics['avg_msg'][name] = get_avg_msg(dataset[name])
        metrics['sent_receive_ratio'][name] = get_sent_receive_ratio(dataset[name])
        metrics['sent_receive_word_ratio'][name] = get_sent_receive_word_ratio(dataset[name])
        metrics['responsiveness'][name] = get_responsiveness(dataset[name])
        metrics['avg_polarity'][name] = get_avg_polarity(dataset[name], name)
    for name in metrics['avg_msg']:
        metrics['overall'][name] = get_overall_friendship_score(name)
    fix_scale('responsiveness')
    fix_scale('overall')

# change the scale to max out at 1
def fix_scale(metric_name):
    global metrics
    max_value = max(metrics[metric_name].values())
    for name in metrics[metric_name]:
        metrics[metric_name][name] = round(metrics[metric_name][name]/max_value,2)

# store metrics to csv
def store_metric_results_csv():
    with open('metrics_summary.csv','w') as outfile:
        outfile.write('name,avg_msg,sent_receive_ratio,sent_receive_word_ratio,responsiveness,avg_polarity,overall_score\n')
        for name in dataset:
            outfile.write('{},{},{},{},{},{},{}\n'.format(
                name,
                metrics['avg_msg'][name],
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
    plt.xticks(y_pos, map(lambda x: x.lower(),sorted_names))
    plt.title(title)
    plt.ylabel(y_axis)
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('{}.png'.format(metric_name), dpi=200)
    plt.cla()
    plt.clf()
    plt.close()

def main():
    print 'loading dataset'
    load_dict()
    #pickel_dataset()
    print 'analyzing dataset'
    analyze(dataset)

    print 'saving metric results'
    store_metric_results_csv()

    friend_ranking = sorted(metrics['overall'], key=metrics['overall'].get, reverse=True)
    best_5 = friend_ranking[:5]
    worst_5 = friend_ranking[-5:]
    plot_polarity_trend_multi(best_5,'best')
    plot_polarity_trend_multi(worst_5,'worst')
    
    top_words_heatmap(10)

    plot_metric('avg_polarity', 'Avg polarity of messages. 1 being positive, -1 being negative', 'polarity')
    plot_metric('avg_msg', 'Avg number of messages per day during friendship', 'messages')
    plot_metric('sent_receive_ratio', 'Ratio of messages sent by me to messages received\n (high number is bad)', 'ratio')
    plot_metric('responsiveness', 'Responsiveness (higher number means quicker to respond)', 'responsiveness')
    plot_metric('overall', 'Overall friendship score', 'how much you mean to me according\n to this program')
    plot_metric('sent_receive_word_ratio', 'Ratio of non-stopwords sent by me to non-stopwords received\n (high number is bad)', 'ratio')
    
if __name__ == "__main__":
    main()
