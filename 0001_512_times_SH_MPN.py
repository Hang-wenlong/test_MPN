# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 23:23:19 2020

@author: hp
"""


import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import pickle
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
import pandas as pd
import numpy as np

import time
import torch
import copy
from shutil import copyfile
import re

# torch.cuda.set_device(0)

# =============================================================================
class tiles_splidate(Dataset):
    def __init__(self, X_Y_frame, transform=None):
        self.X_Y_frame = X_Y_frame 
        self.transform = transform

    def __len__(self):
        return  self.X_Y_frame.shape[0]

    def __getitem__(self, index):
        fn,label,sample_id = self.X_Y_frame.iloc[index,:]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, label,sample_id


def extract_num(x):
    temp_ = os.path.split(x)
    temp_ = temp_[1].split('_')
    return '_'.join(temp_[:2])

def get_X_Y_frame(Case_addr,control_addr):
    case_group = get_slide_path(Case_addr)
    cont_group = get_slide_path(control_addr)
    X_Y_frame = pd.DataFrame(np.transpose(np.array(cont_group + case_group )))
    X_Y_frame.columns = ['File_addr']
    X_Y_frame['GT_tiles_level'] = [0]*len(cont_group) + [1]*len(case_group)
    X_Y_frame['Sample_id'] = X_Y_frame['File_addr'].apply(extract_num)
    return X_Y_frame


from sklearn import metrics
def Find_Optimal_Cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations
    predicted : Matrix with predicted data, where rows are observations
    Returns
    -------     
    list type, with optimal cutoff value
    """
    fpr, tpr, threshold = metrics.roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 
                        'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]] 
    return roc_t['threshold']



def perf_measure(y_actual, y_hat):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_hat)): 
        if y_actual[i]==y_hat[i]==1:
           TP += 1
        if y_hat[i]==1 and y_actual[i]!=y_hat[i]:
           FP += 1
        if y_actual[i]==y_hat[i]==0:
           TN += 1
        if y_hat[i]==0 and y_actual[i]!=y_hat[i]:
           FN += 1 
    Sen = TP/(TP+FN)
    Spe = TN/(FP+TN)
    return(Sen,Spe)


# def predict_softmax(inputs,model,device):
#     model.to(device)
#     with torch.no_grad():
#         inputs = inputs.to(device)
#         out = model(inputs)
#         pre = torch.softmax(out, 1) 
#     return(pre[:,1].cpu().tolist())

import copy
# df = copy.deepcopy(hist_final)
def clean_hist(df):
    block_key = list(df.keys())
    Sub_key = list(df[block_key[0]].keys())
    out_frame = pd.DataFrame(np.zeros((len(df[block_key[0]][Sub_key[0]]),
                                       int(len(block_key)*len(Sub_key)))))
    Col_num = []
    Col_index = 0
    for block_key_i in block_key:
        for Sub_key_i in Sub_key:
            out_frame.iloc[:,Col_index] = df[block_key_i][Sub_key_i]
            Col_index = Col_index+1
            Col_num.append('_'.join([block_key_i,Sub_key_i]))

    out_frame.columns = Col_num
    return out_frame


# model_fit = torch.load( r'F:\Phase1_Module\model\Baselinemodel\Baseline_resnet18.pkl')


# # res_test = predict_classification(model_fit,Valid_frame,device)
# res_test = predict_classification(model_fit,Train_frame,device)

# res_test['labels'] = res_test['labels'].apply(lambda x:int(x))
# res_test['tile_prediction'] = res_test['tile_prediction'].apply(lambda x:float(x)) 

# fpr, tpr, thresholds = roc_curve(res_test['labels'],
#                                  res_test['tile_prediction'] )
# AUC_tiles = auc(fpr, tpr)
# print(AUC_tiles)

def predict_classification(model_fit,X_Y_frame,device):
    model_fit = model_fit.to(device)
    model_fit.eval()
    Data_set = tiles_splidate(X_Y_frame ,
                                transforms.Compose([
                                transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.7471,0.6316,0.7629], 
                                                     [0.2271,0.2782,0.1806])]))
    data_generater = DataLoader(Data_set, 
                              batch_size=32, 
                              shuffle=False)

 
    tile_predict,tiles_nums,labels_out = [],[],[]
    for inputs ,labels,tiles_numi in data_generater:
        inputs = inputs.to(device) 
        outputs = model_fit(inputs) 
        softmax_out = torch.softmax(outputs, 1)
        tile_predict = tile_predict + softmax_out[:,1].tolist() 
        
        # tiles_nums = tiles_nums+tiles_numi.cpu().tolist()
        tiles_nums = tiles_nums+list(tiles_numi) 
        labels_out = labels_out + labels.cpu().tolist()

    temp_data_frame = np.array([tiles_nums,labels_out,
                                tile_predict])
    temp_data_frame = pd.DataFrame(np.transpose(temp_data_frame))
    temp_data_frame.columns = ['tile_num',
                               'labels',
                               "tile_prediction"] 
    return temp_data_frame
    

# =============================================================================
 
# prepare X_y_frame

os.chdir(r'/data/MPN')

file_all = []
for root_i ,dirs_i,files_i in os.walk('./Data/Scaled_raw_512'):
    for f_i in files_i:
        file_all.append(os.path.join(root_i, f_i))

temp_addr = np.array(file_all)
X_Y_ALL = pd.DataFrame(np.transpose( temp_addr))
X_Y_ALL.columns = ['File_addr_temp']

X_Y_ALL["key_id"] = X_Y_ALL["File_addr_temp"].apply(lambda x:os.path.split(x)[1])

GT_temp = pd.read_csv("/data/MPN/Data/GT_file.csv")


Final_res_all_new = X_Y_ALL.merge(GT_temp, 
                             left_on ="key_id" ,
                             right_on ="File_addr",
                             how = "right")

Final_res_all_new["File_addr"] = Final_res_all_new["File_addr_temp"] 
# 0 helth 1 reactive 2 MF 3 PV 4 ET


Final_res_all_new["GT_tiles_level"].value_counts()

for i in range(Final_res_all_new.shape[0]):
    if Final_res_all_new.loc[i,"GT_tiles_level"] <2:
        Final_res_all_new.loc[i,"GT_tiles_level"] =0
    else:
        Final_res_all_new.loc[i,"GT_tiles_level"] =Final_res_all_new.loc[i,"GT_tiles_level"] -1
	

Final_res_all_new["GT_tiles_level"].value_counts()



Train_frame = Final_res_all_new[Final_res_all_new['Data_source']=='Train']
# Train_frame = Train_frame[Train_frame['GT_tiles_level']!=2]
Train_frame = Train_frame.reset_index(drop=True)
Train_frame = Train_frame.loc[:,['File_addr', 'GT_tiles_level', 'Sample_id']]

Test_frame = Final_res_all_new[Final_res_all_new['Data_source']=='Test']
# Test_frame = Test_frame[Test_frame['GT_tiles_level']!=2]
Test_frame = Test_frame.reset_index(drop=True)
Test_frame = Test_frame.loc[:,['File_addr', 'GT_tiles_level', 'Sample_id']]

Valid_frame = Final_res_all_new[Final_res_all_new['Data_source']=='Valid']
# Valid_frame = Valid_frame[Valid_frame['GT_tiles_level']!=2]
Valid_frame = Valid_frame.reset_index(drop=True)
Valid_frame = Valid_frame.loc[:,['File_addr', 'GT_tiles_level', 'Sample_id']]


 
            

data_trans = { 'train':transforms.Compose([
                # transforms.Resize(1024),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.7471,0.6316,0.7629], 
                                     [0.2271,0.2782,0.1806])
            ]),
                'test':transforms.Compose([
                # transforms.Resize(1024),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.7471,0.6316,0.7629], 
                                     [0.2271,0.2782,0.1806])
            ]),
               'valid':transforms.Compose([
                # transforms.Resize(1024),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.7471,0.6316,0.7629], 
                                     [0.2271,0.2782,0.1806])
            ]) }


image_data_set= {'train':tiles_splidate(Train_frame,
                                        data_trans['train']),
                 'test': tiles_splidate(Test_frame,
                                       data_trans['test']),
                 'valid': tiles_splidate(Valid_frame,
                                       data_trans['valid'])
                 }

dataloaders =  {'train':DataLoader(image_data_set['train'], 
                          batch_size=32, 
                          shuffle=True),
                 'test': DataLoader(image_data_set['test'], 
                          batch_size=32, 
                          shuffle=False),
                 'valid': DataLoader(image_data_set['valid'], 
                          batch_size=32, 
                          shuffle=False)}
 
dataset_sizes = {x: len(image_data_set[x]) for x in ['train', 'test','valid']}

 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



from torchvision import datasets, models, transforms
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.metrics import f1_score

# model = model_ft
# optimizer = optimizer_ft
# scheduler = exp_lr_scheduler 
# num_epochs=25
 
def train_model(model, criterion, optimizer, scheduler, num_epochs=25):
    since = time.time()
    Train_his = {'train':{"loss_tiles":[], 
                          "tiles_F1micro":[],
                          "tiles_F1macro":[],
                          "tiles_F1weighted":[],
                          "S_F1micro":[],
                          "S_F1macro":[],
                          "S_F1weighted":[] },
                 'test':{"loss_tiles":[], 
                          "tiles_F1micro":[],
                          "tiles_F1macro":[],
                          "tiles_F1weighted":[],
                          "S_F1micro":[],
                          "S_F1macro":[],
                          "S_F1weighted":[]},
                 'valid':{"loss_tiles":[], 
                          "tiles_F1micro":[],
                          "tiles_F1macro":[],
                          "tiles_F1weighted":[],
                          "S_F1micro":[],
                          "S_F1macro":[],
                          "S_F1weighted":[]}} 
    best_model_wts = copy.deepcopy(model.state_dict())
    min_loss = 100

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 60) 
        # Each epoch has a training and validation phase
        for phase in ['train', 'test','valid']:
            
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode 
            running_loss = 0.0
            # running_corrects = 0
             
            Pred_cal = []
            labels_cal = []
            sample_id_cal =[] 
            # Iterate over data.
            for inputs, labels,sample_id_list in dataloaders[phase]:
                # print("a")
                inputs = inputs.to(device) 
                labels = labels.to(device)
                # break 
                optimizer.zero_grad()
                labels_cal = labels_cal + labels.tolist()
                sample_id_cal = sample_id_cal + list(sample_id_list)
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    softmax_out = torch.softmax(outputs, 1)
                    softmax_out_temp = softmax_out.cpu().detach().numpy() 
                    softmax_out_temp = pd.DataFrame(softmax_out_temp)
                    list_res = softmax_out_temp.apply(lambda x: np.argmax(x),axis=1)
                      
                    Pred_cal = Pred_cal + list(list_res)
                    loss = criterion(outputs, labels.long()) 
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                
 
            hist_res = pd.DataFrame(np.transpose(np.array([Pred_cal  ,
                                                           labels_cal])))
            hist_res['Sample_id'] = sample_id_cal
            hist_res.columns = ['Pred','GT','Sample_id']
            hist_res['GT'] = hist_res['GT'].apply(lambda x:int(x)) 
            hist_res['Pred'] = hist_res['Pred'].apply(lambda x:int(x)) 
            
            hist_res_sample = pd.DataFrame(hist_res.groupby(['Sample_id',"GT"])['Pred'].median())
            hist_res_sample["GT"] =  list(map(lambda x: x[1], hist_res_sample.index))   
            hist_res_sample['GT'] = hist_res_sample['GT'].apply(lambda x:int(x)) 
            hist_res_sample['Pred'] = hist_res_sample['Pred'].apply(lambda x:int(x)) 
            

  
            
            if phase == 'train':
                scheduler.step()
 
            epoch_loss = running_loss / dataset_sizes[phase]  
            tilesmicro = f1_score(list(hist_res['GT']),list(hist_res['Pred']) ,average='micro'  )
            tilesmacro = f1_score(list(hist_res['GT']),list(hist_res['Pred']) ,average='macro'  )
            tilesweight = f1_score(list(hist_res['GT']),list(hist_res['Pred']) ,average='weighted'  )
            # hist_res['Pred'].value_counts(dropna=False)
            
            Ssmicro =f1_score(list(hist_res_sample['GT']),list(hist_res_sample['Pred']) ,average='micro' )
            Ssmacro = f1_score(list(hist_res_sample['GT']),list(hist_res_sample['Pred']) ,average='macro'  )
            Ssweight =f1_score(list(hist_res_sample['GT']),list(hist_res_sample['Pred']) ,average='weighted'  ) 
            
            
            Train_his[phase]["loss_tiles"].append(epoch_loss)
            Train_his[phase]["tiles_F1micro"].append(tilesmicro)
            Train_his[phase]["tiles_F1macro"].append(tilesmacro)
            Train_his[phase]["tiles_F1weighted"].append(tilesweight)
            
            Train_his[phase]["S_F1micro"].append(Ssmicro)
            Train_his[phase]["S_F1macro"].append(Ssmacro)
            Train_his[phase]["S_F1weighted"].append(Ssweight)       
            
            print('{} Loss: {:.4f} tilesmicro: {:.4f} tilesmacro: {:.4f} tilesweight: {:.4f} Ssmicro: {:.4f} Ssmacro: {:.4f} Ssweight: {:.4f}'.format(
                phase, epoch_loss, 
                tilesmicro,tilesmacro,tilesweight,
                Ssmicro,Ssmacro,Ssweight))
 
            # deep copy the model
            if phase == 'test' and epoch_loss < min_loss:
                min_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
        print()
 
        
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    # print('max f score: {:4f}'.format(max_fscore))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model,Train_his  

 
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 4) 
model_ft = model_ft.to(device)
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)


model_ss,hist_final = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)

# model = model_ft

torch.save(model_ss,
           r'/data/MPN/results/SH_MF_PV_ET_512_times.pkl')

his_csv_out = clean_hist(hist_final)
his_csv_out.to_csv(r'/data/MPN/results/SH_MF_PV_ET_512_times.csv')
            
















