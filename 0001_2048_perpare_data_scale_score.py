# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 22:01:03 2020

@author: hp
"""


import pandas as pd
# import numpy as np
import os
import re
from Phase1_module_2.Try_fliter_only_code.fliter_only_utils import *

import random

def unique_sample(x):
    temp0 =  os.path.split(x) 
    temp1 =  os.path.split(temp0[0])
    temp2 =  temp0[1].split('.')
    return "_".join([temp1[1],temp2[0]])
    
All_Raw_file  = get_filelist(path=r'G:\MPN\MPN_Phase1_raw_data')

# 311



All_Raw_file['unique_sample'] = "1"
for i in range(All_Raw_file.shape[0]):
    All_Raw_file['unique_sample'][i] =  unique_sample(All_Raw_file.iloc[i,0])
    if re.search("sub_health", All_Raw_file.iloc[i,0]) is not None:
        All_Raw_file['unique_sample'][i] = "S"+All_Raw_file['unique_sample'][i]
 
All_Raw_file = All_Raw_file[  All_Raw_file['unique_sample']!= 'S01_1M1708'] 

All_Raw_file=All_Raw_file.reset_index(drop=True)


base_addr = r'G:\MPN\Phase1_module_2\Try_fliter_only_data2048\Temp_Scale_summary_tiles'

test_whole = Slide_combine_fliter(RAW_SLIDE_FILE_LIST= All_Raw_file.iloc[:,0],
                                  unique_id = All_Raw_file.iloc[:,1],
                                  BASE_ADDR = base_addr,
                                  TILE_SIZE = 2048)

## step1 scale
test_whole.Scale_slide()   
## step2 fliter mask
Scale_Raw_file  = get_slide_path(r'G:\MPN\Phase1_module_2\Try_fliter_only_data2048\Temp_Scale_summary_tiles\scale')
test_whole.fliter_slide(Scale_Raw_file)  
# slide_path_list = Scale_Raw_file

Filter_Raw_file = get_slide_path(r'G:\MPN\Phase1_module_2\Try_fliter_only_data2048\Temp_Scale_summary_tiles\filter')
## step3 tile scores
test_whole.Tiles_save( Filter_Raw_file ,save_tiles=False) 

Summary_tiles_Raw_file  = get_slide_path(r'G:\MPN\Phase1_module_2\Try_fliter_only_data2048\Temp_Scale_summary_tiles\tiles_summary')

All_Raw_file.columns = ["Raw_files","unique_sample_id"]
All_Raw_file['Scale_files'] = Scale_Raw_file
All_Raw_file['Fileter_files'] = Filter_Raw_file
All_Raw_file['Tiles_summary']  = Summary_tiles_Raw_file
Label_file = pd.read_csv(r'G:\MPN\MPN_Phase1_raw_data/label_new.csv',
                         encoding='gbk')  

Label_file['文件夹'] = Label_file['文件夹'].apply(lambda x: str(x) )
Label_file['pre_sample'] = Label_file['文件夹'].apply(lambda x: x.zfill(2) )
Label_file['unique_sample_id'] = Label_file.loc[:,['pre_sample','文件名']].apply(lambda x:"_".join(x),
                                                                              axis=1)

Final_label = pd.merge(All_Raw_file, Label_file, on='unique_sample_id',how='left')

Final_label["GT"] = Final_label['诊断'].apply(lambda x:int(x!='未见明显异常'))

for i in range(Final_label.shape[0]):
    if re.search('sub_health', Final_label['Raw_files'][i]) is not None:
         Final_label["GT"][i]= 2

# Final_label['GT'].value_counts()


# Case
Case_all = Final_label[Final_label["GT"]==1]
Case_all=Case_all.sample(frac=1.0,random_state=10 )
Case_all.reset_index(drop=True,inplace=True)
# control
Control_all = Final_label[Final_label["GT"]==0]
Control_all=Control_all.sample(frac=1.0,random_state=10 )
Control_all.reset_index(drop=True,inplace=True)

# sub_health
sub_health_all = Final_label[Final_label["GT"]==2]
sub_health_all=sub_health_all.sample(frac=1.0,random_state=10 )
sub_health_all.reset_index(drop=True,inplace=True)




# 1/2  1/4  1/4
Training_part = create_path(base_dir=r'G:\MPN\Phase1_module_2\Try_fliter_only_data2048',
                            file_name = "Train")
Test_part = create_path(base_dir=r'G:\MPN\Phase1_module_2\Try_fliter_only_data2048',
                            file_name = "Test")
Validation_part = create_path(base_dir=r'G:\MPN\Phase1_module_2\Try_fliter_only_data2048',
                            file_name = "Valid")

# add new fold
if os.path.exists('G:\MPN\Phase1_module_2\Try_fliter_only_data2048\Train\sub_health')==False:
    os.makedirs('G:\MPN\Phase1_module_2\Try_fliter_only_data2048\Train\sub_health' )
if os.path.exists('G:\MPN\Phase1_module_2\Try_fliter_only_data2048\Test\sub_health')==False:
    os.makedirs('G:\MPN\Phase1_module_2\Try_fliter_only_data2048\Test\sub_health' )
if os.path.exists('G:\MPN\Phase1_module_2\Try_fliter_only_data2048\Valid\sub_health')==False:
    os.makedirs('G:\MPN\Phase1_module_2\Try_fliter_only_data2048\Valid\sub_health' )


import random
###########################################  case
# Training part
for i in Case_all.iloc[:110,:].index:
    print(Case_all['Raw_files'][i])
    get_tiles_res( Case_all['Raw_files'][i] ,
                  Case_all['Tiles_summary'][i],
                  seed_in=i*100+10,
                  extrect_num=20000,
                  unique_sample_id = Case_all['unique_sample_id'][i],
                  tiles_out_addr = Training_part[0])

# Testing part
for i in Case_all.iloc[110:165,:].index:
    print(Case_all['Raw_files'][i])
    get_tiles_res( Case_all['Raw_files'][i] ,
                  Case_all['Tiles_summary'][i],
                  seed_in=i*100+10,
                  extrect_num=20000,
                  unique_sample_id = Case_all['unique_sample_id'][i],
                  tiles_out_addr = Test_part[0])

# Validation part
for i in Case_all.iloc[165:,:].index:
    print(Case_all['Raw_files'][i])
    get_tiles_res( Case_all['Raw_files'][i] ,
                  Case_all['Tiles_summary'][i],
                  seed_in=i*100+10,
                  extrect_num=20000,
                  unique_sample_id = Case_all['unique_sample_id'][i],
                  tiles_out_addr = Validation_part[0])



###########################################  control
# Training part
for i in Control_all.iloc[:37,:].index:
    print(Control_all['Raw_files'][i])
    get_tiles_res( Control_all['Raw_files'][i] ,
                  Control_all['Tiles_summary'][i],
                  seed_in=i*100+20,
                  extrect_num=20000,
                  unique_sample_id = Control_all['unique_sample_id'][i],
                  tiles_out_addr = Training_part[1])

# Testing part
for i in Control_all.iloc[37:55,:].index:
    print(Control_all['Raw_files'][i])
    get_tiles_res( Control_all['Raw_files'][i] ,
                  Control_all['Tiles_summary'][i],
                  seed_in=i*100+20,
                  extrect_num=20000,
                  unique_sample_id = Control_all['unique_sample_id'][i],
                  tiles_out_addr = Test_part[1])

# Validation part
for i in Control_all.iloc[55:,:].index:
    print(Control_all['Raw_files'][i])
    get_tiles_res( Control_all['Raw_files'][i] ,
                  Control_all['Tiles_summary'][i],
                  seed_in=i*100+20,
                  extrect_num=20000,
                  unique_sample_id = Control_all['unique_sample_id'][i],
                  tiles_out_addr = Validation_part[1])






###########################################  sub_health
# Training part
for i in sub_health_all.iloc[:50,:].index:
    print(sub_health_all['Raw_files'][i])
    get_tiles_res( sub_health_all['Raw_files'][i] ,
                  sub_health_all['Tiles_summary'][i],
                  seed_in=i*100+20,
                  extrect_num=20000,
                  unique_sample_id = sub_health_all['unique_sample_id'][i],
                  tiles_out_addr = 'G:\MPN\Phase1_module_2\Try_fliter_only_data2048\Train\sub_health')

# Testing part
for i in sub_health_all.iloc[50:75,:].index:
    print(sub_health_all['Raw_files'][i])
    get_tiles_res( sub_health_all['Raw_files'][i] ,
                  sub_health_all['Tiles_summary'][i],
                  seed_in=i*100+20,
                  extrect_num=20000,
                  unique_sample_id = sub_health_all['unique_sample_id'][i],
                  tiles_out_addr ='G:\MPN\Phase1_module_2\Try_fliter_only_data2048\Test\sub_health')

# Validation part
for i in sub_health_all.iloc[75:,:].index:
    print(sub_health_all['Raw_files'][i])
    get_tiles_res( sub_health_all['Raw_files'][i] ,
                  sub_health_all['Tiles_summary'][i],
                  seed_in=i*100+20,
                  extrect_num=20000,
                  unique_sample_id = sub_health_all['unique_sample_id'][i],
                  tiles_out_addr = 'G:\MPN\Phase1_module_2\Try_fliter_only_data2048\Valid\sub_health')















 
