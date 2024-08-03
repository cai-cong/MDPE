# -*- coding: utf-8 -*-
import torch.nn as nn
import torch.optim as optim
import os
import torch
import torch.nn as nn
from sklearn.metrics import f1_score,accuracy_score,precision_score,recall_score
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
from sklearn.metrics import roc_auc_score,confusion_matrix
import sys
#训练模型
def train_model(model, trainloader,devloader,testloader,args):
    epochs, lr, modelname= args.epochs, args.lr, args.log_file_name
    if not os.path.exists(os.path.join("./model", modelname)):
        os.makedirs(os.path.join("./model", modelname))
    optimizer = optim.Adam(model.parameters(), lr=lr)    #Adam优化器
    best_val_rmse = float('inf')
    # best_test_rmse = float('inf')
    #best_test_emse = float('inf')
    print('Start training')
    patience = 20
    train_lossssss = 999999
    for epoch in range(1, epochs + 1):
        train_loss = train(model, trainloader, optimizer,args)
        if train_loss <train_lossssss:
            train_lossssss = train_loss
            print("------------------------",train_lossssss)
            patience = 20
        if patience == 0:
            print("-------------------------------早停了")
            break
        patience = patience -1
        if args.task=="emotion" or (args.task == "deception" and args.regerssion==True):
            val_rmse = evaluate(model, devloader,args)
            # test_rmse = evaluate(model, testloader,args)
            print('-' * 50)
            print(f'Epoch:{epoch:>3} | [Train] | Loss: {train_loss:>.3f}')
            print(f'Epoch:{epoch:>3} | [Val]   | MSE : {val_rmse:>.3f} ')     #输出训练和验证loss
            #print(f'Epoch:{epoch:>3} | [Test]  | MSE : {test_rmse:>.3f} ')     #输出训练和验证loss
            print('-' * 50)

            if val_rmse<best_val_rmse:
                best_val_rmse = val_rmse
                print(f"Best val rmse: {best_val_rmse}")
                # torch.save(model,f"{modelname}.pth")
        if args.task=="deception" and args.regerssion==False:
            acc,jianchu,xujing,f1,auc = evaluate(model, devloader,args)
            print(f'Epoch:{epoch:>3} | [Train] | Loss: {train_loss:>.3f}')
            print(f"准确率:{acc:.5f},检出率:{jianchu:.5f},虚警率:{xujing:.5f},f1:{f1:.5f},auc:{auc:.5f}")
    # print(f"Best test rmse: {best_test_rmse}")


#每轮训练模型
def train(model, trainloader, optimizer,args):
    model.train()
    running_loss = 0.
    if args.task == "emotion" or (args.task == "deception" and args.regerssion==True):
        criterion = torch.nn.MSELoss()
    if args.task == "deception" and args.regerssion==False:
        criterion = torch.nn.CrossEntropyLoss()
    # criterion = torch.nn.MSELoss()
    for i, train_data in enumerate(trainloader):
        optimizer.zero_grad()
        if args.fusion==True:
            if args.AVT ==True:
                if args.use_personality==True and args.use_emotion==False:
                    features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3,personality = train_data
                    personality = personality.cuda()
                    features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3 = \
                    features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2,features3.cuda(), labels3.cuda(), lengths3
                    predictions = model.forward([features1,features2,features3,personality], [labels, lengths2],args)
                elif args.use_personality==False and args.use_emotion==True:
                    features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3,emotion = train_data
                    emotion = emotion.cuda()
                    features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3 = \
                    features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2,features3.cuda(), labels3.cuda(), lengths3
                    predictions = model.forward([features1,features2,features3,emotion], [labels, lengths2],args)
                elif args.use_personality==True and args.use_emotion==True:
                    features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3,emotion,personality = train_data
                    emotion = emotion.cuda()
                    personality = personality.cuda()
                    features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3 = \
                    features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2,features3.cuda(), labels3.cuda(), lengths3
                    predictions = model.forward([features1,features2,features3,emotion,personality], [labels, lengths2],args)
                else:
                    features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3 = train_data
                    features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3 = \
                    features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2,features3.cuda(), labels3.cuda(), lengths3
                    predictions = model.forward([features1,features2,features3], [labels, labels2,labels3],args)
            else:
                if args.use_personality==True and args.use_emotion==False:
                    features1, labels, lengths1,features2, labels2, lengths2,personality = train_data
                    personality = personality.cuda()
                    features1, labels, lengths1,features2, labels2, lengths2 = \
                    features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2
                    predictions = model.forward([features1,features2,personality], [labels, lengths2],args)
                elif args.use_emotion==True and args.use_personality==False:
                    features1, labels, lengths1,features2, labels2, lengths2,emotion = train_data
                    emotion = emotion.cuda()
                    features1, labels, lengths1,features2, labels2, lengths2 = \
                    features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2
                    predictions = model([features1,features2,emotion], [labels, lengths2],args)
                elif args.use_emotion==True and args.use_personality==True:
                    features1, labels, lengths1,features2, labels2, lengths2,emotion,personality = train_data
                    emotion = emotion.cuda()
                    personality = personality.cuda()
                    features1, labels, lengths1,features2, labels2, lengths2 = \
                    features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2
                    predictions = model([features1,features2,emotion,personality], [labels, lengths2],args)    
                else:
                    features1, labels, lengths1,features2, labels2, lengths2 = train_data
                    features1, labels, lengths1,features2, labels2, lengths2 = \
                    features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2
                    predictions = model([features1,features2], [labels, lengths2],args)
        else:
            if args.use_personality==True and args.use_emotion==False:
                features, labels, lengths,personality = train_data
                personality = personality.cuda()
                features, labels = features.cuda(), labels.cuda()
                predictions = model.forward([features,personality],lengths,args)
            if args.use_emotion==True and args.use_personality==False:
                # if args.use_emotion
                features, labels, lengths,emotion = train_data
                features, labels = features.cuda(), labels.cuda()
                emotion = emotion.cuda()
                predictions = model([features,emotion],lengths,args)
            if args.use_emotion==True and args.use_personality==True:
                # if args.use_emotion
                features, labels, lengths,emotion,personality = train_data
                features, labels = features.cuda(), labels.cuda()
                emotion = emotion.cuda()
                personality = personality.cuda()
                predictions = model([features,emotion,personality],lengths,args)
            if args.use_emotion==False and args.use_personality==False:
                features, labels, lengths = train_data
                features, labels = features.cuda(), labels.cuda()
                predictions = model.forward(features,lengths,args)
        #print(predictions)
        if args.task == "emotion" or (args.task == "deception" and args.regerssion==True):
            criterion = torch.nn.MSELoss()
            if len(labels.shape) == 1:
                labels = labels.unsqueeze(1)
            loss = criterion(predictions, labels.float())
        elif args.task == "deception" and args.regerssion==False:
            loss = criterion(predictions, (labels > 0).long())

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(trainloader)


def evaluate(model, val_loader,args):
    model.eval()
    if args.task == "emotion" or (args.task == "deception" and args.regerssion==True):
        with torch.no_grad():
            predictions_corr = np.empty((0, args.classnum))
            labels_corr = np.empty((0, args.classnum))
            for i, val_data in enumerate(val_loader):
                if args.fusion==True:
                    if args.AVT ==True:
                        if args.use_personality==True:
                            features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3,personality = val_data
                            personality = personality.cuda()
                            features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3 = \
                            features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2,features3.cuda(), labels3.cuda(), lengths3
                            predictions = model.forward([features1,features2,features3,personality], [labels, lengths2],args)
                        else:
                            features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3 = val_data
                            features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3 = \
                            features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2,features3.cuda(), labels3.cuda(), lengths3
                            predictions = model.forward([features1,features2,features3], [labels, labels2,labels3],args)
                    else:
                        if args.use_personality==True:
                            features1, labels, lengths1,features2, labels2, lengths2,personality = val_data
                            personality = personality.cuda(0)
                            features1, labels, lengths1,features2, labels2, lengths2 = \
                            features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2
                            predictions = model.forward([features1,features2,personality], [labels, lengths2],args)
                        else:
                            features1, labels, lengths1,features2, labels2, lengths2 = val_data
                            features1, labels, lengths1,features2, labels2, lengths2 = \
                            features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2
                            predictions = model.forward([features1,features2], [labels, lengths2],args)
                else:
                    if args.use_personality==True:
                        features, labels, lengths,personality = val_data
                        personality = personality.cuda()
                        features, labels = features.cuda(), labels.cuda()
                        predictions = model.forward([features,personality],lengths,args)
                    elif args.use_emotion==True:
                        features, labels, lengths,emotion = val_data
                        features, labels = features.cuda(), labels.cuda()
                        emotion = emotion.cuda()
                        predictions = model.forward([features,emotion],lengths,args)
                    else:
                        features, labels, lengths = val_data
                        features, labels = features.cuda(), labels.cuda()
                        predictions = model.forward(features,lengths,args)
                predictions = predictions.cpu().numpy()
                if len(labels.shape)==1:
                    labels = labels.unsqueeze(1)
                labels = labels.cpu().numpy()

                predictions_corr = np.append(predictions_corr, predictions, axis=0)
                labels_corr = np.append(labels_corr, labels, axis=0)

            labels_corr = labels_corr
            predictions_corr = predictions_corr
            rmse = sqrt(mean_squared_error(predictions_corr, labels_corr))
        return rmse
    else:
        with torch.no_grad():
            # predictions_corr = np.empty((0, 1))
            # labels_corr = np.empty((0, 1))
            pre_list, label_list, pro_list = [], [], []
            import torch.nn as nn
            softmax = nn.Softmax(dim=1)
            for i, val_data in enumerate(val_loader):
                if args.fusion==True:
                    if args.AVT ==True:
                        if args.use_personality==True and args.use_emotion==False:
                            features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3,personality = val_data
                            personality = personality.cuda()
                            features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3 = \
                            features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2,features3.cuda(), labels3.cuda(), lengths3
                            predictions = model.forward([features1,features2,features3,personality], [labels, lengths2],args)
                        elif args.use_personality==False and args.use_emotion==True:
                            features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3,emotion = val_data
                            emotion = emotion.cuda()
                            features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3 = \
                            features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2,features3.cuda(), labels3.cuda(), lengths3
                            predictions = model.forward([features1,features2,features3,emotion], [labels, lengths2],args)
                        elif args.use_personality==True and args.use_emotion==True:
                            features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3,emotion,personality = val_data
                            emotion = emotion.cuda()
                            personality = personality.cuda()
                            features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3 = \
                            features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2,features3.cuda(), labels3.cuda(), lengths3
                            predictions = model.forward([features1,features2,features3,emotion,personality], [labels, lengths2],args)
                        else:
                            features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3 = val_data
                            features1, labels, lengths1,features2, labels2, lengths2,features3, labels3, lengths3 = \
                            features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2,features3.cuda(), labels3.cuda(), lengths3
                            predictions = model.forward([features1,features2,features3], [labels, labels2,labels3],args)
                    else:
                        if args.use_personality==True and args.use_emotion==False:
                            features1, labels, lengths1,features2, labels2, lengths2,personality = val_data
                            personality = personality.cuda()
                            features1, labels, lengths1,features2, labels2, lengths2 = \
                            features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2
                            predictions = model.forward([features1,features2,personality], [labels, lengths2],args)
                        elif args.use_emotion==True and args.use_personality==False:
                            features1, labels, lengths1,features2, labels2, lengths2,emotion = val_data
                            emotion = emotion.cuda()
                            features1, labels, lengths1,features2, labels2, lengths2 = \
                            features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2
                            predictions = model.forward([features1,features2,emotion], [labels, lengths2],args)
                        elif args.use_emotion==True and args.use_personality==True:
                            features1, labels, lengths1,features2, labels2, lengths2,emotion,personality = val_data
                            emotion = emotion.cuda()
                            personality = personality.cuda()
                            features1, labels, lengths1,features2, labels2, lengths2 = \
                            features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2
                            predictions = model.forward([features1,features2,emotion,personality], [labels, lengths2],args)
                        elif args.use_emotion==False and args.use_personality==False:
                            features1, labels, lengths1,features2, labels2, lengths2 = val_data
                            features1, labels, lengths1,features2, labels2, lengths2 = \
                            features1.cuda(), labels.cuda(), lengths1,features2.cuda(), labels2.cuda(), lengths2
                            predictions = model.forward([features1,features2], [labels, lengths2],args)
                else:
                    if  args.use_personality==True and args.use_emotion==False:
                        features, labels, lengths,personality = val_data
                        personality = personality.cuda()
                        features, labels = features.cuda(), labels.cuda()
                        predictions = model.forward([features,personality],lengths,args)
                    elif args.use_emotion==True and args.use_personality==False:
                        features, labels, lengths,emotion = val_data
                        features, labels = features.cuda(), labels.cuda()
                        emotion = emotion.cuda()
                        predictions = model.forward([features,emotion],lengths,args)
                    elif args.use_emotion==True and args.use_personality==True:
                            features1, labels, lengths1,emotion,personality = val_data
                            emotion = emotion.cuda()
                            personality = personality.cuda()
                            features1, labels, lengths1 = \
                            features1.cuda(), labels.cuda(), lengths1
                            predictions = model.forward([features1,emotion,personality], [labels, lengths1],args)
                    else:
                        features, labels, lengths = val_data
                        features, labels = features.cuda(), labels.cuda()
                        predictions = model.forward(features,lengths,args)
                labels = (labels > 0).float()
                # print(predictions)
                # print(torch.argmax(predictions,dim=1).tolist())
                # print(labels)
                pre_list += torch.argmax(predictions,dim=1).tolist()
                label_list += labels.tolist()#
                pro_list += softmax(predictions).tolist()
            # print(label_list)
            # print(np.sum(np.array(label_list)))
            acc = accuracy_score(pre_list,label_list)
            f1 = f1_score(pre_list,label_list)
            auc = roc_auc_score(np.array(label_list),np.array(pro_list)[:,1])
            confusion = confusion_matrix(pre_list,label_list)

            jianchu = confusion[1][1]/(confusion[0][1]+confusion[1][1])
            xujing = confusion[1][0]/(confusion[0][0]+confusion[1][0])
        return acc,jianchu,xujing,f1,auc
