# -*- coding: utf-8 -*-
import argparse
import torch.nn as nn
from dataset import get_dataloader
from train import train_model
from model import Model
import os
from dateutil import tz
from datetime import datetime
import sys
from dataset_load import load_data
import torch
import numpy as np
import random


class Logger(object):
    def __init__(self, log_file="log_file.log"):
        self.terminal = sys.stdout
        self.file = open(log_file, "w")

    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.file.flush()

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--parallel', action='store_true',default=False,help='whether use DataParallel')
    #baichuan13B-base 5120 chatglm2-6B 4096 chinese-hubert-base 768 chinese-hubert-large 1024 chinese-wav2vec2-base 768
    # clipVIT-B16 512 clipVIT-L14 768 sbert-chinese-general-v2 768 VIT 768 wavlm-base 768 wavlm-large 1024
    parser.add_argument('--feature_set', default="baichuan13B-base", type=str)
    parser.add_argument('--fea_dim', default=5120, type=int)
    parser.add_argument('--fusion',action='store_true', default=True)
    parser.add_argument('--AVT',action='store_true', default=True)
    parser.add_argument('--feature_set2', default="clipVIT-B16", type=str)
    parser.add_argument('--fea_dim2', default=512, type=int)
    parser.add_argument('--feature_set3', default="chinese-hubert-base", type=str)
    parser.add_argument('--fea_dim3', default=768, type=int)

    parser.add_argument('--task', default="deception", type=str)# deception  emotion  
    # parser.add_argument('--dataset_file_path', default="/mnt/data1/caicong/data/release/emotion/")
    parser.add_argument('--dataset_file_path', default="/data2/release/deception_features/")
    parser.add_argument('--use_personality',action='store_true', default=True)
    parser.add_argument('--use_emotion',action='store_true', default=True)
    parser.add_argument('--regerssion',action='store_true', default=False)
    parser.add_argument('--classnum', default=2, type=int)
    # parser.add_argument('--patience', default=50, type=int)
    parser.add_argument('--epochs', default=300, type=int, help='number of training epochs')
    parser.add_argument('--batch_size', default=64, type=int, help='size of a mini-batch')
    parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')

    args = parser.parse_args()
    if not os.path.exists("./log"):
        os.makedirs("./log")
    args.log_file_name = '{}_[{}_{}]_[{}_{}]'.format(
        datetime.now(tz=tz.gettz()).strftime("%Y-%m-%d-%H-%M"), args.feature_set, args.fea_dim,args.lr, args.batch_size)
    sys.stdout = Logger(os.path.join("./log", args.log_file_name + '.log'))
    print(args.feature_set)
    print("AVT:",args.AVT)
    print("fusion:",args.fusion)
    print("use_personality:",args.use_personality)
    print("use_emotion:",args.use_emotion)

    data = load_data(args)
    if args.fusion==True:
        t= args.fea_dim
        args.feature_set = args.feature_set2
        args.fea_dim = args.fea_dim2
        data2 = load_data(args)
        if args.AVT==True:
            args.feature_set = args.feature_set3
            args.fea_dim = args.fea_dim3
            data3 = load_data(args)
            data = [data,data2,data3]
        else:
            data = [data,data2]
        args.fea_dim = t
    print(args.feature_set)

    train_dataloader, dev_dataloader, test_dataloader = get_dataloader(args,data)
    model = Model(args)

    if args.parallel:
        model = nn.DataParallel(model)

    model = model.cuda()
    print('=' * 50)

    train_model(model, train_dataloader,dev_dataloader, test_dataloader, args)

    print('=' * 50)

if __name__ == "__main__":
    main()