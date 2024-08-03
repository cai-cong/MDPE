import pandas
import math
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import numpy as np
import json
import glob
from torch.utils.data import DataLoader

class EmotionDataset(Dataset):
    def __init__(self, partition, data,args=None):
        self.fusion=False
        self.use_personality=False
        self.use_emotion=False
        if args !=None and args.fusion==True:
            self.use_personality = False
            self.fusion = args.fusion
            data1 = data[0]
            data2 = data[1]
            if args.use_personality==True:
                self.use_personality = args.use_personality
                self.personality = data1[partition]['personality']
            if args.use_emotion==True:
                self.use_emotion = args.use_emotion
                self.emotion = data1[partition]['emotion']
            features1, labels1 = data1[partition]['feature'], data1[partition]['label']
            self.features1 = features1
            self.labels1 = labels1
            features2, labels2 = data2[partition]['feature'], data2[partition]['label']
            self.features2 = features2
            self.labels2 = labels2
            self.AVT = args.AVT
            if self.AVT==True:
                data3 = data[2]
                features3, labels3 = data3[partition]['feature'], data3[partition]['label']
                self.features3 = features3
                self.labels3 = labels3
        else:
            features1, labels1 = data[partition]['feature'], data[partition]['label']
            if args!=None and args.use_personality==True and args.use_emotion==False:
                self.use_personality = args.use_personality
                self.personality = data[partition]['personality']
            if args!=None and args.use_emotion==True and args.use_personality==False:
                self.use_emotion = args.use_emotion
                self.emotion = data[partition]['emotion']
            if args!=None and args.use_emotion==True and args.use_personality==True:
                self.use_personality = args.use_personality
                self.personality = data[partition]['personality']
                self.use_emotion = args.use_emotion
                self.emotion = data[partition]['emotion']
            self.features1 = features1
            self.labels1 = labels1

    def __getitem__(self, idx):
        if self.use_emotion==True:
            emotion = self.emotion[idx]
        if self.use_personality==True:
            personality = self.personality[idx]
        if self.fusion==True:
            feature1 = self.features1[idx]
            label1 = self.labels1[idx]
            feature2 = self.features2[idx]
            label2 = self.labels2[idx]
            if self.AVT == True:
                feature3 = self.features3[idx]
                label3 = self.labels3[idx]
                if self.use_emotion==True and self.use_personality==False:
                    return feature1, label1,feature2, label2,feature3, label3,emotion
                elif self.use_personality==True and self.use_emotion==False:
                    return feature1, label1,feature2, label2,feature3, label3,personality
                elif self.use_personality==True and self.use_emotion==True:
                    return feature1, label1,feature2, label2,feature3, label3,emotion,personality
                else:
                    return feature1, label1,feature2, label2,feature3, label3
            else:
                if self.use_emotion==True and self.use_personality==False:
                    return feature1, label1,feature2, label2,emotion
                elif self.use_personality==True and self.use_emotion==False:
                    return feature1, label1,feature2, label2,personality
                elif self.use_personality==True and self.use_emotion==True:
                    return feature1, label1,feature2, label2,emotion,personality
                else:
                    return feature1, label1,feature2, label2
        else:
            feature1 = self.features1[idx]
            label1 = self.labels1[idx]
            if self.use_personality==True and self.use_emotion==False:
                return feature1, label1,personality
            elif self.use_emotion==True and self.use_personality==False:
                return feature1, label1,emotion
            elif self.use_emotion==True and self.use_personality==True:
                return feature1, label1,emotion,personality
            else:
                return feature1, label1

    def __len__(self):
        return len(self.features1)

def collate_fn(data):
    if len(data[0])==8:
        data1 = [(item[0],item[1]) for item in data]
        data2 = [(item[2],item[3]) for item in data]
        data3 = [(item[4],item[5]) for item in data]
        fuzhu1 = [item[6] for item in data]
        fuzhu2 = [item[7] for item in data]
        for i in range(0,len(data1)):
            data1[i] = (i,data1[i])
        data1.sort(key=lambda x:x[1][0].shape[0],reverse=True)

        ids1, tmp_features_labels1 = zip(*data1)
        features_tmp1,labels1 = zip(*tmp_features_labels1)
        labels1 = torch.FloatTensor(np.array(labels1))

        feature_dim1 = len(features_tmp1[0][0])
        lengths1 = [len(f) for f in features_tmp1]
        max_length1 = max(lengths1)
        features1 = torch.zeros((len(features_tmp1), max_length1,  feature_dim1)).float()

        for i ,feature in enumerate(features_tmp1):
            end = lengths1[i]
            feature = torch.FloatTensor(np.array(feature))
            features1[i,:end, :]= feature[:end, :]
            
        for i in range(0,len(data2)):
            data2[i] = (i,data2[i])
        data2.sort(key=lambda x:x[1][0].shape[0],reverse=True)

        ids2, tmp_features_labels2 = zip(*data2)
        features_tmp2,labels2 = zip(*tmp_features_labels2)
        labels2 = torch.FloatTensor(np.array(labels2))

        feature_dim2 = len(features_tmp2[0][0])
        lengths2 = [len(f) for f in features_tmp2]
        max_length2 = max(lengths2)
        features2 = torch.zeros((len(features_tmp2), max_length2,  feature_dim2)).float()

        for i ,feature in enumerate(features_tmp2):
            end = lengths2[i]
            feature = torch.FloatTensor(np.array(feature))
            features2[i,:end, :]= feature[:end, :]
        for i in range(0,len(data3)):
            data3[i] = (i,data3[i])
        data3.sort(key=lambda x:x[1][0].shape[0],reverse=True)
        ids3, tmp_features_labels3 = zip(*data3)
        features_tmp3,labels3 = zip(*tmp_features_labels3)
        labels3 = torch.FloatTensor(np.array(labels3))
        feature_dim3 = len(features_tmp3[0][0])
        lengths3 = [len(f) for f in features_tmp3]
        max_length3 = max(lengths3)
        features3 = torch.zeros((len(features_tmp3), max_length3,  feature_dim3)).float()

        for i ,feature in enumerate(features_tmp3):
            end = lengths3[i]
            feature = torch.FloatTensor(np.array(feature))
            features3[i,:end, :]= feature[:end, :]
        return features1,labels1, lengths1,features2,labels2, lengths2,features3,labels3, lengths3,torch.tensor(np.array(fuzhu1),dtype=torch.float32),torch.tensor(np.array(fuzhu2),dtype=torch.float32)
    #3个特征两个辅助是8 三个特征1个辅助是7 6的话就是两个特征两个辅助 要不是三个特征
    if len(data[0])==6 or len(data[0])==7:
        data1 = [(item[0],item[1]) for item in data]
        data2 = [(item[2],item[3]) for item in data]
        data3 = [(item[4],item[5]) for item in data]
        #两个辅助，两个特征
        if len(data3[0][1].shape)!=0 and len(data3[0][0])==128 and len(data3[0][1])==60 and len(data[0])==6:
            fuzhu = [item[4] for item in data]
            fuzhu2 = [item[5] for item in data]
        #一个辅助3个特征
        if len(data[0])==7:
            fuzhu = [(item[6]) for item in data]
        for i in range(0,len(data1)):
            data1[i] = (i,data1[i])
        data1.sort(key=lambda x:x[1][0].shape[0],reverse=True)

        ids1, tmp_features_labels1 = zip(*data1)
        features_tmp1,labels1 = zip(*tmp_features_labels1)
        labels1 = torch.FloatTensor(np.array(labels1))

        feature_dim1 = len(features_tmp1[0][0])
        lengths1 = [len(f) for f in features_tmp1]
        max_length1 = max(lengths1)
        features1 = torch.zeros((len(features_tmp1), max_length1,  feature_dim1)).float()

        for i ,feature in enumerate(features_tmp1):
            end = lengths1[i]
            feature = torch.FloatTensor(np.array(feature))
            features1[i,:end, :]= feature[:end, :]
            
        for i in range(0,len(data2)):
            data2[i] = (i,data2[i])
        data2.sort(key=lambda x:x[1][0].shape[0],reverse=True)

        ids2, tmp_features_labels2 = zip(*data2)
        features_tmp2,labels2 = zip(*tmp_features_labels2)
        labels2 = torch.FloatTensor(np.array(labels2))

        feature_dim2 = len(features_tmp2[0][0])
        lengths2 = [len(f) for f in features_tmp2]
        max_length2 = max(lengths2)
        features2 = torch.zeros((len(features_tmp2), max_length2,  feature_dim2)).float()

        for i ,feature in enumerate(features_tmp2):
            end = lengths2[i]
            feature = torch.FloatTensor(np.array(feature))
            features2[i,:end, :]= feature[:end, :]
        #3个特征
        import copy
        datax = copy.deepcopy(data3)
        if len(data3[0][1].shape)==0 and isinstance(data3[0][1], np.int64) and (len(data[0])==6 or len(data[0])==7):
            for i in range(0,len(data3)):
                data3[i] = (i,data3[i])
            data3.sort(key=lambda x:x[1][0].shape[0],reverse=True)
            ids3, tmp_features_labels3 = zip(*data3)
            features_tmp3,labels3 = zip(*tmp_features_labels3)
            labels3 = torch.FloatTensor(np.array(labels3))
            feature_dim3 = len(features_tmp3[0][0])
            lengths3 = [len(f) for f in features_tmp3]
            max_length3 = max(lengths3)
            features3 = torch.zeros((len(features_tmp3), max_length3,  feature_dim3)).float()

            for i ,feature in enumerate(features_tmp3):
                end = lengths3[i]
                feature = torch.FloatTensor(np.array(feature))
                features3[i,:end, :]= feature[:end, :]
        if len(data[0])==7:
            return features1,labels1, lengths1,features2,labels2, lengths2,features3,labels3, lengths3,torch.tensor(np.array(fuzhu),dtype=torch.float32)
        elif len(datax[0][0])==128 and len(datax[0][1])==60 and len(data[0])==6:
            return features1,labels1, lengths1,features2,labels2, lengths2,torch.tensor(np.array(fuzhu),dtype=torch.float32),torch.tensor(np.array(fuzhu2),dtype=torch.float32)
        else:
            return features1,labels1, lengths1,features2,labels2, lengths2,features3,labels3, lengths3

    elif len(data[0])==4 or len(data[0])==5:
        data1 = [(item[0],item[1]) for item in data]
        data2 = [(item[2],item[3]) for item in data]
        #两个辅助一个特征
        if len(data[0])==4 and len(data2[0][0])==128 and len(data2[0][1].shape)!=0 and len(data2[0][1])==60 :
            fuzhu =  [(item[2]) for item in data]
            fuzhu2 = [(item[3]) for item in data]
        if len(data[0])==5:#一个辅助两个特征
            fuzhu = [(item[4]) for item in data]
        for i in range(0,len(data1)):
            data1[i] = (i,data1[i])
        data1.sort(key=lambda x:x[1][0].shape[0],reverse=True)

        ids1, tmp_features_labels1 = zip(*data1)
        features_tmp1,labels1 = zip(*tmp_features_labels1)
        labels1 = torch.FloatTensor(np.array(labels1))

        feature_dim1 = len(features_tmp1[0][0])
        lengths1 = [len(f) for f in features_tmp1]
        max_length1 = max(lengths1)
        features1 = torch.zeros((len(features_tmp1), max_length1,  feature_dim1)).float()

        for i ,feature in enumerate(features_tmp1):
            end = lengths1[i]
            feature = torch.FloatTensor(np.array(feature))
            features1[i,:end, :]= feature[:end, :]
            #两个特征
        import copy
        datax = copy.deepcopy(data2)
        if len(data2[0][1].shape)==0 and isinstance(data2[0][1], np.int64) and (len(data[0])==4 or len(data[0])==5):
            for i in range(0,len(data2)):
                data2[i] = (i,data2[i])
        
            data2.sort(key=lambda x:x[1][0].shape[0],reverse=True)

            ids2, tmp_features_labels2 = zip(*data2)
            features_tmp2,labels2 = zip(*tmp_features_labels2)
            labels2 = torch.FloatTensor(np.array(labels2))

            feature_dim2 = len(features_tmp2[0][0])
            lengths2 = [len(f) for f in features_tmp2]
            max_length2 = max(lengths2)
            features2 = torch.zeros((len(features_tmp2), max_length2,  feature_dim2)).float()

            for i ,feature in enumerate(features_tmp2):
                end = lengths2[i]
                feature = torch.FloatTensor(np.array(feature))
                features2[i,:end, :]= feature[:end, :]
        if len(data[0])==5:
            return features1,labels1, lengths1,features2,labels2,lengths2,torch.tensor(np.array(fuzhu),dtype=torch.float32)
        elif len(datax[0][1].shape)!=0 and len(datax[0][0])==128 and len(datax[0][1])==60 and len(data[0])==4:
            return features1,labels1, lengths1,torch.tensor(np.array(fuzhu),dtype=torch.float32),torch.tensor(np.array(fuzhu2),dtype=torch.float32)
        else:
            return features1,labels1, lengths1,features2,labels2,lengths2
    else:
        if len(data[0])==3:
            emotion = [(item[2]) for item in data]
            data1 = [(item[0],item[1]) for item in data]
        else:
            data1 = data
        for i in range(0,len(data1)):
            data1[i] = (i,data1[i])
        data1.sort(key=lambda x:x[1][0].shape[0],reverse=True)

        ids, tmp_features_labels = zip(*data1)
        features_tmp,labels = zip(*tmp_features_labels)
        labels = torch.FloatTensor(np.array(labels))

        feature_dim = len(features_tmp[0][0])
        lengths = [len(f) for f in features_tmp]
        max_length = max(lengths)
        features = torch.zeros((len(features_tmp), max_length,  feature_dim)).float()

        for i ,feature in enumerate(features_tmp):
            end = lengths[i]
            feature = torch.FloatTensor(np.array(feature))
            features[i,:end, :]= feature[:end, :]
        if len(data[0])==3:
            return features,labels, lengths,torch.tensor(np.array(emotion),dtype=torch.float32)
        else:
            return features,labels, lengths

def get_dataloader(args,data):
    batch_size = args.batch_size
    train_dataset = EmotionDataset("train", data,args)
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size,num_workers=4, shuffle=True,collate_fn=collate_fn)

    dev_dataset  = EmotionDataset("val", data,args)
    dev_dataloader = DataLoader(dataset=dev_dataset,batch_size=batch_size,num_workers=4, shuffle=False,collate_fn=collate_fn)

    test_dataset  = EmotionDataset("val", data,args)
    test_dataloader = DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=4, shuffle=False,collate_fn=collate_fn)

    return train_dataloader, dev_dataloader, test_dataloader