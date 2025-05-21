import numpy as np
import pandas as pd
import os.path as osp
import numpy.linalg as nlina


def roadDataMSbyCSV(root,scs, tg, Cs_end, Cu_start):
    Xs,Ys,l=[],[],[]
    li=0
    for sc in scs:  
        source = np.array(pd.read_csv(osp.join(root,sc), header=None))
        source = source[source[:, -1] < Cs_end, :]  # 返回bool值作为行索引,选出符合数据
        print('source {} is {}, size:{}'.format(li,sc,source.shape))
        Xsi = source[:, :-1]
        Ysi = source[:, -1].astype(int)
        Xsi = Xsi / nlina.norm(Xsi, axis=1, keepdims=True)
        Xs.append(Xsi)
        Ys.append(Ysi)
        l=np.hstack((np.array(l),np.full((source.shape[0],),li)))
        li+=1

    target = np.array(pd.read_csv(osp.join(root,tg), header=None))

    C = np.size(np.unique(source[:, -1]))
    Cs = Cs_end
    Cu = C - Cu_start

    if Cs_end != Cu_start:
        target = np.vstack((target[target[:, -1] < Cs_end, :], target[target[:, -1] >= Cu_start, :]))
    Xt = target[:, :-1]
    Yt = target[:, -1].astype(int)
    Xt = Xt / nlina.norm(Xt, axis=1, keepdims=True)
    print('target {} is {}, size:{}'.format(li,tg,target.shape))
    
    l=np.hstack((np.array(l),np.full((Xt.shape[0],),li)))

    return Xs, Ys, Xt, Yt,l,Cs, Cu


# scs=['Product.csv','RealWorld.csv','Art.csv']
# tg='Clipart.csv'
# root=osp.join('Office-31_Alex','Data_office31')
# 以下是officeHome
root='data/OfficeHome_dk'
dataSet='OfficeHomeDK'
domain=['Product.csv','RealWorld.csv','Art.csv','Clipart.csv']
Cs_end,Cu_start=25,25

## 以下是office31
# root='data/resnet50Data'
# dataSet='Office31'
# domain=['dslr.csv','amazon.csv','webcam.csv']
# # Cs_end,Cu_start=10,20
# Cs_end,Cu_start=20,20

for tg in domain:
    print("Current target:{}".format(tg))
    t=domain.copy()
    t.remove(tg)
    scs=t
    Xs,ys,Xt,yt,l,Cs,Cu=roadDataMSbyCSV(root,scs,tg,Cs_end,Cu_start)
    print("-"*20)