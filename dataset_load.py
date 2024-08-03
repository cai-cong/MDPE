import glob
import os
import time
import numpy as np
import pandas as pd
import pickle
import random

def get_data_partition(partition_file):
    vid2partition, partition2vid = {}, {}
    df = pd.read_csv(partition_file)

    for row in df.values:
        vid, partition = str("%03d"%row[0]), row[1]
        vid2partition[vid] = partition
        if partition not in partition2vid:
            partition2vid[partition] = []
        if vid not in partition2vid[partition]:
            partition2vid[partition].append(vid)
    return vid2partition, partition2vid
import re

def extract_number(filename):
    # 使用正则表达式匹配中间的数字
    match = re.search(r'-(\d+)-', filename)
    if match:
        return int(match.group(1))  # 返回匹配的数字
    else:
        return 0  # 如果没有匹配到数字，则返回 0
def load_data(args):
    feature_path = os.path.join(args.dataset_file_path,"features",args.feature_set)
    label_path = os.path.join(args.dataset_file_path,"labels")

    data_file_name = f'data_{args.feature_set}_{args.fea_dim}.pkl'
    data_file = os.path.join(f'./data_cache/{args.task}/', data_file_name)
    if args.use_personality ==True and args.use_emotion==False:
        data_file = os.path.join(f'./data_cache/personality_{args.task}/', data_file_name)
    elif args.use_emotion==True and args.use_personality ==False:
        data_file = os.path.join(f'./data_cache/emotion/', data_file_name)
    elif args.use_emotion==True and args.use_personality ==True:
        data_file = os.path.join(f'./data_cache/emotion_personality/', data_file_name)
    else:
        data_file = os.path.join(f'./data_cache/{args.task}/', data_file_name)
    if os.path.exists(data_file) and args.use_emotion==True:  # check if file of preprocessed data exists
        print(f'Find cached data "{os.path.basename(data_file)}".')
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        return data
    elif os.path.exists(data_file) and args.use_emotion==False:
        print(f'Find cached data "{os.path.basename(data_file)}".')
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
        return data

    print('Constructing data from scratch ...')
    data = {'train': {'feature': [], 'label': []},
            'val': {'feature': [], 'label': []},
            'test': {'feature': [], 'label': []}}
    if args.use_personality==True and args.use_emotion==False:
        data['train']['personality']=[]
        data['val']['personality']=[]
        data['test']['personality']=[]
        personality = pd.read_csv(os.path.join(args.dataset_file_path,"personality.csv"))
    if args.use_emotion==True and args.use_personality==False:
        data['train']['emotion']=[]
        data['val']['emotion']=[]
        data['test']['emotion']=[]
        emotion = pd.read_csv(os.path.join(args.dataset_file_path,"emotion.csv"), header=None)
    if args.use_emotion==True and args.use_personality==True:
        data['train']['personality']=[]
        data['val']['personality']=[]
        data['test']['personality']=[]
        personality = pd.read_csv(os.path.join(args.dataset_file_path,"personality.csv"))
        data['train']['emotion']=[]
        data['val']['emotion']=[]
        data['test']['emotion']=[]
        emotion = pd.read_csv(os.path.join(args.dataset_file_path,"emotion.csv"), header=None)
    for vid in sorted(os.listdir(feature_path)):
        lie = 2
        nolie = 3
        label_file = os.path.join(label_path, vid + '.csv')
        label = (pd.read_csv(label_file).iloc[:, 3]).to_numpy()
        files = os.listdir(os.path.join(feature_path,vid))
        files = sorted(files, key=extract_number)    
        random.seed(int(vid))
        random.shuffle(files) #乱序遍历文件夹
        for file in files:
            number = int(file.split('-')[1])
            if file.endswith("csv"):
                feature = pd.read_csv(os.path.join(feature_path, vid, file), header=None).to_numpy()
            elif file.endswith("npy"):
                feature = np.load(os.path.join(feature_path, vid, file))
            if  label[number-1]>0 and  lie>0:
                lie = lie - 1
                data["val"]['label'].append(label[number - 1])
                data["val"]['feature'].append(feature)
                if args.use_personality==True and args.use_emotion==False:
                    personality_data = personality.iloc[:,1:61].to_numpy()[int(vid)-1]
                    data["val"]['personality'].append(personality_data)
                if args.use_emotion==True and args.use_personality==False:
                    emotion_data = emotion.to_numpy()[int(vid)-1]
                    data["val"]['emotion'].append(emotion_data)
                if args.use_emotion==True and args.use_personality==True:
                    personality_data = personality.iloc[:,1:61].to_numpy()[int(vid)-1]
                    data["val"]['personality'].append(personality_data)
                    emotion_data = emotion.to_numpy()[int(vid)-1]
                    data["val"]['emotion'].append(emotion_data)
            elif  label[number-1]==0 and nolie>0:
                nolie = nolie - 1
                data["val"]['label'].append(label[number - 1])
                data["val"]['feature'].append(feature)
                if args.use_personality==True and args.use_emotion==False:
                    personality_data = personality.iloc[:,1:61].to_numpy()[int(vid)-1]
                    data["val"]['personality'].append(personality_data)
                if args.use_emotion==True and args.use_personality==False:
                    emotion_data = emotion.to_numpy()[int(vid)-1]
                    data["val"]['emotion'].append(emotion_data)
                if args.use_emotion==True and args.use_personality==True:
                    personality_data = personality.iloc[:,1:61].to_numpy()[int(vid)-1]
                    data["val"]['personality'].append(personality_data)
                    emotion_data = emotion.to_numpy()[int(vid)-1]
                    data["val"]['emotion'].append(emotion_data)
            else:
                data["train"]['label'].append(label[number - 1])
                data["train"]['feature'].append(feature)
                if args.use_personality==True and args.use_emotion==False:
                    personality_data = personality.iloc[:,1:61].to_numpy()[int(vid)-1]
                    data["train"]['personality'].append(personality_data)
                if args.use_emotion==True and args.use_personality==False:
                    emotion_data = emotion.to_numpy()[int(vid)-1]
                    data["train"]['emotion'].append(emotion_data)
                if args.use_emotion==True and args.use_personality==True:
                    personality_data = personality.iloc[:,1:61].to_numpy()[int(vid)-1]
                    data["train"]['personality'].append(personality_data)
                    emotion_data = emotion.to_numpy()[int(vid)-1]
                    data["train"]['emotion'].append(emotion_data)
        print(vid)
    print(len(data["train"]['label']))
    print(len(data["val"]['label']))
    print("数据加载完成")
    if args.task=="deception":
        if args.use_personality==True: 
            if not os.path.exists("./data_cache/personality_deception"):
                os.mkdir("./data_cache/personality_deception")
        else:
            if not os.path.exists("./data_cache/deception"):
                os.mkdir("./data_cache/deception")
    elif args.task=="emotion":
        if args.use_personality==True: 
            if not os.path.exists("./data_cache/personality_emotion"):
                os.mkdir("./data_cache/personality_emotion")
        else:
            if not os.path.exists("./data_cache/emotion"):
                os.mkdir("./data_cache/emotion")
    pickle.dump(data, open(data_file, 'wb'))

    return data