# -*- coding: utf-8 -*-
"""
Created on Tue Dec 15 17:01:39 2020

@author: dell
"""


import os
import shutil
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


# new_path = 'D:/论文code/MPN/To_Feng/bag_data'

# os.chdir(r'D:\论文code\MPN\To_Feng')
# GT_temp = pd.read_csv("GT_file.csv")

# print(GT_temp["GT_tiles_level"].value_counts())

# GT_temp.loc[GT_temp["GT_tiles_level"] <2, ["GT_tiles_level"]] = 0
# GT_temp.loc[GT_temp["GT_tiles_level"] >=2, ["GT_tiles_level"]] = GT_temp["GT_tiles_level"] -1



# file_all = []

# for root_i ,dirs_i,files_i in os.walk('./Scaled_raw'):
#     for f_i in files_i:
#         file_all.append(os.path.join(root_i, f_i))
        
# for i in range(GT_temp.shape[0]):
#     if GT_temp.loc[i,"GT_tiles_level"] == 3:
#         shutil.copy(file_all[i], os.path.join(new_path, str(3)))
#     # elif GT_temp.loc[i,"GT_tiles_level"] == 1:
#     #     shutil.copy(file_all[i], os.path.join(new_path, str(1)))
#     # elif GT_temp.loc[i,"GT_tiles_level"] == 2:
#     #     shutil.copy(file_all[i], os.path.join(new_path, str(2)))     
#     # else GT_temp.loc[i,"GT_tiles_level"] == 3:
#     #     shutil.copy(file_all[i], os.path.join(new_path, str(3)))
#########################################################################################
# new_path = 'D:/论文code/MPN/To_Feng/bag_data/0'

# file_all = []
# patient_all = []
# for root_i ,dirs_i,files_i in os.walk('./bag_data/3'):
#     for f_i in files_i:
#         each_path = os.path.join(root_i, f_i)
#         each_path = each_path.replace('\\','/') ## 替换为D:\\图片\\Zbtv1.jpg
#         file_all.append(each_path)
#         fa = each_path.split('/')[-1] ##取出每个bag的标签
#         patient = fa.split('_')[0] + fa.split('_')[1]
#         patient_all.append(patient)
# # for i in range(GT_temp.shape[0]):
# #     if GT_temp.loc[i,"GT_tiles_level"] == 3:
# #         shutil.copy(file_all[i], os.path.join(new_path, str(3)))
# #     # elif GT_temp.loc[i,"GT_tiles_level"] == 1:
# #     #     shutil.copy(file_all[i], os.path.join(new_path, str(1)))
# #     # elif GT_temp.loc[i,"GT_tiles_level"] == 2:
# #     #     shutil.copy(file_all[i], os.path.join(new_path, str(2)))     
# #     # else GT_temp.loc[i,"GT_tiles_level"] == 3:
# #     #     shutil.copy(file_all[i], os.path.join(new_path, str(3)))
os.chdir(r'D:\论文code\MPN\To_Feng')


 
file_all = []
for root_i ,dirs_i,files_i in os.walk('./Scaled_raw'):
    for f_i in files_i:
        file_all.append(os.path.join(root_i, f_i))

temp_addr = np.array(file_all)
X_Y_ALL = pd.DataFrame(np.transpose( temp_addr))
X_Y_ALL.columns = ['File_addr_temp']

X_Y_ALL["key_id"] = X_Y_ALL["File_addr_temp"].apply(lambda x:os.path.split(x)[1])

GT_temp = pd.read_csv("GT_file.csv")

GT_temp.loc[GT_temp["GT_tiles_level"] <2, ["GT_tiles_level"]] = 0
GT_temp.loc[GT_temp["GT_tiles_level"] >=2, ["GT_tiles_level"]] = GT_temp["GT_tiles_level"] -1

Final_res_all_new = X_Y_ALL.merge(GT_temp, 
                             left_on ="key_id" ,
                             right_on ="File_addr",
                             how = "right")

Final_res_all_new["File_addr"] = Final_res_all_new["File_addr_temp"] 


ori_path = 'D:/论文code/MPN/To_Feng/bag_data/'
patient_all = Final_res_all_new["Sample_id"].unique()
# 转化为字典类型
# print("字典类型: ", dict(Final_res_all_new["Sample_id"].value_counts()))

# # # 对其中的键值对进行遍历
for key, value in Final_res_all_new["Sample_id"].value_counts().items():
    print("Key: ", key, " value: ", value)
    per_patient = Final_res_all_new[Final_res_all_new["Sample_id"]==key] ##每个病人的所有图片patch
    print(per_patient["GT_tiles_level"].unique())
    if per_patient["GT_tiles_level"].unique()==3:
        newpath = os.path.join(ori_path, str(3)) + '/' + key
        if not os.path.exists(newpath):
            os.makedirs(newpath)
        for i in range(value):
            shutil.copy(per_patient.iloc[i]["File_addr"], newpath)
    # elif per_patient["GT_tiles_level"].unique()==1:
    #     newpath = os.path.join(ori_path, str(1)) + '/' + key
    #     if not os.path.exists(newpath):
    #         os.makedirs(newpath)
    #     for i in range(value):
    #         shutil.copy(per_patient.iloc[i]["File_addr"], newpath)
    # elif per_patient["GT_tiles_level"].unique()==2:
    #     newpath = os.path.join(ori_path, str(2)) + '/' + key
    #     if not os.path.exists(newpath):
    #         os.makedirs(newpath)
    #     for i in range(value):
    #         shutil.copy(per_patient.iloc[i]["File_addr"], newpath)
    # else per_patient["GT_tiles_level"].unique()==3:
    #     newpath = os.path.join(ori_path, str(3)) + '/' + key
    #     if not os.path.exists(newpath):
    #         os.makedirs(newpath)
    #     for i in range(value):
    #         shutil.copy(per_patient.iloc[i]["File_addr"], newpath)   
# print(GT_temp.groupby(["Sample_id"])[patient_all[0]].unique())
# # for per_patient in range(patient_all.shape[0]):
# #     
# #     if not os.path.exists(newpath):
# #         os.makedirs(newpath)
    
    