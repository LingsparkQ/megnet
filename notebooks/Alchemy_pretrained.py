from megnet.models import MEGNetModel
import numpy as np
from operator import itemgetter
import json

from re import U
import pandas as pd
import numpy as np
import time
from glob import glob
import torch
def spatial_dis(a,b):
    return torch.norm(torch.tensor(a)-torch.tensor(b))

neighborlist = {}
for _ in range(11,30,1):
    neighborlist[str(_)]={}
    neighborlist[str(_)]["index1"]=[]
    neighborlist[str(_)]["index2"]=[]
    i=0
    for i in range(_):
        j=0
        while j < _:
            if i != j:
                neighborlist[str(_)]["index1"].append(i)
                neighborlist[str(_)]["index2"].append(j)
            j+=1
    neighborlist[str(_)]["index1"]=tuple(neighborlist[str(_)]["index1"])
    neighborlist[str(_)]["index2"]=tuple(neighborlist[str(_)]["index2"])

graph_all={}
id_list=[]
path='/work/gp23/p23002/AI4Science_ICML/guyuhan/Alchemy/pyg/data-bin/raw/alchemy_data/Alchemy-v20191129/final_version.csv'
df=pd.read_csv(path)                        #读入所有数据
start=time.time()
col_name = df.columns.values                #column name
sort1 = col_name[1]                         #原子数目
sort2 = col_name[0]                         #‘gdb_idx’
df = df.sort_values(by=[sort1,sort2])       #对原子数以及序号进行排序
data={}
data["xyz"]={}
data["atom"]={}
names=locals()
#names 中包含 df9～12 dic9～12 以及 dic 9～12 * （14+2）
# for i in [9,10,11,12]:
#这是一个循环用于处理整个的Alchemy中的重原子数为9～12的数据，如果只处理9的可以注释掉并开启下一行，并且下面的也要同步注释
for i in [9]:
    names['df'+str(i)]=df[df['atom number'].isin([i])]
    names['dic'+str(i)]={}
    #新定义的df9～12 以及 dic9～12
    names['dic'+str(i)]['Z']=[]#Z是原子序数
    names['dic'+str(i)]['Coordinate']=[]#坐标
    names['dic'+str(i)]["Contents"]=[]


search_table = {"H": 1, "B": 5, "C": 6, "N": 7, "O": 8, "F": 9, "Si": 14, "P": 15, "S": 16, "Cl": 17, "Se": 34, "Br": 35}

# for i in [9,10,11,12]:
for i in [9]:
    
    names['delta_dic'+str(i)]={}

    for k in range(len(names['df'+str(i)][col_name[0]])):
        names['dic'+str(i)]['Z']=[]
        names['dic'+str(i)]["Coordinate"]=[]
    #下面这一句是用来测试的
    # for k in range(1):
        if k == 2:
            break
        id = names['df'+str(i)][col_name[0]].iloc[k]
        flag=True
        with open('/work/gp23/p23002/AI4Science_ICML/guyuhan/Alchemy/pyg/data-bin/raw/alchemy_data/Alchemy-v20191129/atom_'
                  +str(i)+'/'+str(id) + ".xyz") as f:
            contents = f.readlines()
            
            for lineID in range(2,len(contents)):
                if contents[lineID].strip().split()[0] =='S' or contents[lineID].strip().split()[0] =='Cl':
                    flag=False
                    
                    
        if flag == True:
            id_list.append(id)
            with open('/work/gp23/p23002/AI4Science_ICML/guyuhan/Alchemy/pyg/data-bin/raw/alchemy_data/Alchemy-v20191129/atom_'
                  +str(i)+'/'+str(id) + ".xyz") as f:
                contents = f.readlines()
                for lineID in range(2, len(contents)):
                    names['dic'+str(i)]["Z"].append(search_table[contents[lineID].strip().split()[0]])
                    names['dic'+str(i)]["Coordinate"].append(list(map(float, contents[lineID].strip().split()[1: ])))
                
            atom_number=(contents[0].strip())
            index1=neighborlist[str(atom_number)]['index1']
            index2=neighborlist[str(atom_number)]['index2']
            bonds=[]
            for m in range(len(index1)):
                bonds.append(spatial_dis(names['dic'+str(i)]["Coordinate"][index1[m]],names['dic'+str(i)]["Coordinate"][index2[m]]))
            bonds=tuple(map(float,bonds))
            graph={"atom":names["dic"+str(9)]["Z"],'bond':bonds,'index1':index1,"index2":index2,'state': [[0, 0]]}
            graph_all[str(id)]=graph
                    


# targets =  ['mu', 'alpha', 'HOMO', 'LUMO', 'gap', 'R2', 'ZPVE', 'U0', 'U', 'H', 'G', 'Cv', 'omega1']
targets =  ['mu', 'alpha']

names=locals()
y_pred = []
for target in targets:
    names[target+"pred1"]=[]
    model = MEGNetModel.from_file('../mvl_models/qm9-2018.6.1/' + target+'.hdf5')
    for id in id_list:
        pred = model.predict_graph(graph_all[str(id)])
        names[target+"pred1"].append(pred)
