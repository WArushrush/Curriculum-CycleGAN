import pickle
from bpemb import BPEmb
import numpy as np
import pickle
languages = ['de', 'en','fr','jp']
lang_dict = {'de': 0, 'en': 1, 'fr': 2, 'jp': 3}
word_dicts = []
domains = ['books','dvd','music']
data_types = ['train.processed','test.processed', 'unlabeled.processed']
root = "/root/data/xiaoyang/PyTorch-CycleGAN-master/cls-acl10-processed"
for lan in languages:
    print("reading ", lan)
    f = open(root+"/"+lan+"_mapped.emb",'r')
    data = f.readlines()
    f.close()
    temp_dict = {}
    for line in data[1:]:
        temp = line.split()
        key = temp[0]
        vec = [float(k) for k in temp[1:]]
        temp_dict[key] = np.array(vec)
    word_dicts.append(temp_dict)
for lan in languages:
    word_dict = word_dicts[lang_dict[lan]]
    f = open(root+"/"+lan+'_stop.txt','r')
    stop_list = f.readlines()
    f.close()
    stop_list = [ss.split('\n')[0] for ss in stop_list]
    print("language: ", lan)
    for domain in domains:
        print("  ", domain)
        for data_type in data_types:
            print("        ", data_type)
            f = open(root + "/" + lan + "/" + domain + "/" + data_type, 'r')
            temp = f.readlines()
            f.close()
            temp = [i.split() for i in temp]
            words = [i[:-1] for i in temp]
            sentiments = [i[-1] for i in temp]
            res_reviews = []
            res_sentiments = []
            for i in range(len(words)):
                line = words[i]
                sentiment = sentiments[i]
                if sentiment.split(":")[1] == 'positive':
                    sentiment = 1
                elif sentiment.split(":")[1] == 'negative':
                    sentiment = 0
                sentence_vec = np.zeros(100)
                word_num = 0
                for word in line:
                    try:
                        word = word.split(':')
                        key = word[0]
                        num = int(word[1])
                        if key in stop_list:
                            continue
                        word_num += num
                        sentence_vec += word_dict[key]*num
                    except Exception as e:
                        pass
                sentence_vec /= word_num
                res_reviews.append(sentence_vec.tolist())
                res_sentiments.append(sentiment)
            f = open(root + "/" + lan + "/" + domain + "/" + data_type.split('.')[0] + ".pkl", 'wb+')
            pickle.dump([res_reviews, res_sentiments], f)
