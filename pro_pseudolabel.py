import numpy as np
import math

def progressive(Xt_new, Yt_pseudo, Yt_conf, t, T, Cs, sign):   #sign=False说明在W前，sign=True说明在W后
    # 1.判断已知类中最多的类别个数,计算需要信任的已知类样本
    trust=np.full((Xt_new.shape[0]),False)

    maxweight_sample=np.zeros((Cs,Xt_new.shape[1]))
    max_known_num=0
    for i in range(0,Cs):
        i_index=np.where(Yt_pseudo==i)[0]
        if max_known_num<i_index.shape[0]:
            max_known_num=i_index.shape[0]
        Yt_conf_i=Yt_conf[i_index]
        max_index=np.argmax(Yt_conf_i,axis=0)
        maxweight_sample[i,:]=Xt_new[i_index[max_index],:]
    i_index=np.where(Yt_pseudo==Cs)[0]
    max_unknown_num=i_index.shape[0]
    unknown_num=math.ceil(max_unknown_num*t/T)
    known_num=math.ceil(max_known_num*t/T)
    class_remain=np.full((Cs+1),known_num)
    class_remain[Cs]=unknown_num
    visit=np.full((Xt_new.shape[0]),False)   #判断是否遍历过
    for i in range(0,Xt_new.shape[0]):
        max_weight=0
        max_index=-1
        for j in range(0,Xt_new.shape[0]):
            if max_weight<Yt_conf[j] and visit[j]==False:
                max_weight=Yt_conf[j]
                max_index=j
        if class_remain[Yt_pseudo[max_index]]>0:
            class_remain[Yt_pseudo[max_index]]-=1
            visit[max_index]=True
            trust[max_index]=True
        else:
            visit[max_index]=True
    
    if sign==False:
        return Yt_pseudo,trust
    

    #计算簇
    sim_array=calculateCosSim(Xt_new,maxweight_sample)
    cluster=np.argmax(sim_array,axis=1)
    for i in range(0,Cs):
        cluster_index=np.where(cluster==i)[0]
        max_weight=0
        max_index=-1
        for j in range(0,cluster_index.shape[0]):
            if Yt_pseudo[cluster_index[j]]==Cs and Yt_conf[cluster_index[j]]>max_weight:
                max_weight=Yt_conf[cluster_index[j]]
                max_index=cluster_index[j]
        maxunk_sample=Xt_new[max_index,:]
        unksim=calculateCosSim(Xt_new[cluster_index,:],maxunk_sample.reshape(1,-1))
        knosim=sim_array[cluster_index,i]
        for j in range(0,knosim.shape[0]):
            if knosim[j]<unksim[j]:
                cluster[cluster_index[j]]=Cs
    
    cluster_index=np.where(cluster==Cs)[0]
    cluster_pseudo=Yt_pseudo[cluster_index]
    trust[cluster_index[np.where(cluster_pseudo==Cs)[0]]]=True

    for i in range(0,Cs):
        i_index=np.where(Yt_pseudo==i)[0]
        cluster_i=cluster[i_index]
        for j in range(0,Cs+1):
            if j==i:
                continue
            dif_index=i_index[np.where(cluster_i==j)[0]]
            if dif_index.shape[0]==0:
                continue
            dif_weight=Yt_conf[dif_index]
            sort_index=np.argsort(dif_weight,axis=0)
            sort_index=sort_index[0:math.ceil(sort_index.shape[0]*1/T)]
            trust[dif_index[sort_index]]=True
            Yt_pseudo[dif_index[sort_index]]=Cs
            #if j==Cs:
            #    Yt_pseudo[dif_index[sort_index]]=Cs

    
    out_degree=30  
    for i in range(0,Cs):
        i_index=np.where(cluster==i)[0]
        #构图
        adj_mat=np.full((i_index.shape[0],out_degree),0)     #类似于链表
        if out_degree>=i_index.shape[0]:
            adj_mat=adj_mat[:,0:i_index.shape[0]-1]
        cluster_sim_array=calculateCosSim(Xt_new[i_index],Xt_new[i_index])
        sort_index=np.argsort(-1*cluster_sim_array,axis=1)
        for line in range(0,sort_index.shape[0]):
            k=out_degree
            c_pos=0
            for column in range(0,sort_index.shape[1]):
                if sort_index[line,column]==line:
                    c_pos+=1
                    continue
                adj_mat[line,column-c_pos]=sort_index[line,column]
                k-=1
                if k==0:
                    break
        #找start
        max_weight=0
        max_index=0
        Yt_pseudo_local=Yt_pseudo[i_index]
        Yt_conf_local=Yt_conf[i_index]
        for j in range(0,i_index.shape[0]):
            if Yt_pseudo_local[j]==Cs and Yt_conf_local[j]>max_weight:
                max_weight=Yt_conf_local[j]
                max_index=j
        start=max_index
        tarvel_index=BFS(adj_mat,start,Yt_pseudo_local,Cs)
        trust[i_index[tarvel_index]]=True

    return Yt_pseudo,trust
