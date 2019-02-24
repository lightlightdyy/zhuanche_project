
import pandas as pd
import numpy as np
import datetime
from pandas import DataFrame

#rawdata=pd.read_csv('10_fe_1.txt',dtype=str,sep='\t',nrows=1000)

passenger_label=pd.read_csv('value_pas_0710.txt',sep='\t',names=['passenger_id','label'],header=None)
jzyh=jiazid.loc[passenger_label['label'] == '3']

raw_pid=list(np.loadtxt('rawdata_pid.txt'))   #761603
pospid=list(passenger_label[passenger_label['label'] == '3'].passenger_id)  # positive pid   18879
negpid=list(passenger_label[passenger_label['label'] == '2'].passenger_id)  # negative  pid 749209

pos_pid=list(set(pospid).intersection(set(raw_pid)))   #18478
neg_pid=list(set(negpid).intersection(set(raw_pid)))    #743124


#--------------------------- begin  提取快车&专车原始数据：正样本&负样本 --------------------
import random
sample_size=int(1.5*len(pos_pid))  
sam_neg_pid=random.sample(neg_pid,sample_size)  #负样本抽取
for i in range(len(pos_pid)):
    pos_pid[i]=str(pos_pid[i])
for i in range(len(sam_neg_pid)):
    sam_neg_pid[i]=str(sam_neg_pid[i])

#找出pid对应正负数据
#专车 neg
given_pid = {p:None for p in sam_neg_pid}
order_file = open("{}/10_fe_1.txt".format('/data1/dengyuying//COMPETING_DATA'), "r")
lines = order_file.readlines()  #readlines读取所有行(直到结束符 EOF)并返回列表
order_file.close()
row_num=[]
num_lines = len(lines)
for i in range(num_lines):
    row = lines[i].strip().split('\t')  #strip()去掉每行头尾空白  
    pid = row[60]#int(row[60])
    if pid in given_pid:  # given_pid:  pos_pid, sam_neg_pid
        row_num.append(i)  #找出pid对应rawdata行号
path_prefix = "{}/neg_data_zc_15.txt".format('/data1/dengyuying//COMPETING_DATA') 
data_file = open(path_prefix, "w")
for i in range(len(row_num)):
    data_file.write(lines[row_num[i]])
data_file.close()
print('Done zhuanche')

#快车  neg
order_file = open("{}/fe_kuaiche.tsv".format('/data1/dengyuying//COMPETING_DATA'), "r")
lines = order_file.readlines()
order_file.close()
row_num=[]
num_lines = len(lines)
for i in range(num_lines):
    row = lines[i].strip().split('\t')
    pid = row[118]#int(row[60])
    if pid in given_pid:  # given_pid:  pos_pid, sam_neg_pid
        row_num.append(i) 
path_prefix = "{}/neg_data_kc_15.txt".format('/data1/dengyuying//COMPETING_DATA') 
data_file = open(path_prefix, "w")
for i in range(len(row_num)):
    data_file.write(lines[row_num[i]])
data_file.close()

#--------------------------- end 提取快车&专车原始数据：正样本&负样本--------------------

#  clean data 

from clean_data_kc_zc import *
path1 = '/nfs/project/dengyuying/zhuanche/neg_data_zc_1102.txt'
path2 = '/nfs/project/dengyuying/zhuanche/neg_data_kc_1102.txt'
file_name1 = 'clean_neg_data_zc_1102.txt'
file_name2 = 'clean_neg_data_kc_1102.txt'
clean_data_kc_zc(path1,path2,file_name1, file_name2)


clean_neg_data_kc = pd.DataFrame(np.loadtxt('/data1/dengyuying/COMPETING_DATA/clean_neg_data_kc_1102.txt'))
clean_neg_data_zc = pd.DataFrame(np.loadtxt('/data1/dengyuying/COMPETING_DATA/clean_neg_data_zc_1102.txt'))
clean_pos_data_kc = pd.DataFrame(np.loadtxt('/data1/dengyuying/COMPETING_DATA/clean_pos_data_kc.txt'))
clean_pos_data_zc = pd.DataFrame(np.loadtxt('/data1/dengyuying/COMPETING_DATA/clean_pos_data_zc.txt'))

# 按pid合并快车与专车数据
clean_pos_merge_kz=clean_pos_data_kc.merge(clean_pos_data_zc,left_on=clean_pos_data_kc.iloc[:,0],right_on=clean_pos_data_zc.iloc[:,0],how="inner")
clean_neg_merge_kz=clean_neg_data_kc.merge(clean_neg_data_zc,left_on=clean_neg_data_kc.iloc[:,0],right_on=clean_neg_data_zc.iloc[:,0],how="inner")

clean_pos_merge_kz=clean_pos_merge_kz.drop(clean_pos_merge_kz.columns[[0,3601]],axis=1) #删除pid
clean_neg_merge_kz=clean_neg_merge_kz.drop(clean_neg_merge_kz.columns[[0,3601]],axis=1)

clean_pos_merge_kz.to_csv('clean_pos_merge_kz_1102.csv')  
clean_neg_merge_kz.to_csv('clean_neg_merge_kz_1102.csv')  


import xgboost as xgb
from xgboost import XGBClassifier
from xgboost import XGBRegressor   # this is for regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,accuracy_score
from scipy.stats import ranksums
from sklearn.metrics import precision_recall_fscore_support
import pickle

clean_pos_merge_kz = pd.read_csv('/data1/dengyuying/COMPETING_DATA/clean_pos_merge_kz_1102.csv')
clean_neg_merge_kz = pd.read_csv('/data1/dengyuying/COMPETING_DATA/clean_neg_merge_kz_1102.csv')
if clean_pos_merge_kz.shape[1] == 5431:
    clean_pos_merge_kz = clean_pos_merge_kz.iloc[:,1:]
if clean_neg_merge_kz.shape[1] == 5431:
    clean_neg_merge_kz = clean_neg_merge_kz.iloc[:,1:]
    
response = [ 1 for _ in range(clean_pos_merge_kz.shape[0])]
response += [0 for _ in range(clean_neg_merge_kz.shape[0])]
y = np.array(response)
x = np.concatenate((clean_pos_merge_kz, clean_neg_merge_kz), axis=0)
from model import *
importances, select_marker, index, train_data = model(y, x)

#半个月专车特征数据 
#clean_pos_merge_kz_half = clean_pos_merge_kz.iloc[:,0:4515]  
#clean_neg_merge_kz_half = clean_neg_merge_kz.iloc[:,0:4515]

#一周专车特征数据
#clean_pos_merge_kz_week = clean_pos_merge_kz.iloc[:,0:4027]  
#clean_neg_merge_kz_week = clean_neg_merge_kz.iloc[:,0:4027]



DAYSINC = 30
sorted_argument = np.argsort(-importances)
inds = index[select_marker[sorted_argument]]
sorted_impo = np.sort(importances)[::-1]

selected_name_kc = np.load("selected_name_kc.npy")
selected_name_zc = np.load("selected_name_zc.npy")
selected_name=np.append(selected_name_kc,selected_name_zc)
all_col_name = []
for i in range(DAYSINC):
    for j in range(len(selected_name)):
        all_col_name.append(str(i+1)+'_'+selected_name[j])
fe_importance = list(zip(sorted_impo, [all_col_name[i] for i in inds]))
fe_importance=DataFrame(fe_importance)
fe_importance[0:20]

