import torch
import numpy as np
import numpy.linalg as nlina
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
import autograd.numpy as anp
import autograd.numpy.linalg as alina
import pymanopt as pm
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers
from scipy import optimize
from tool import roadDataMSbyCSV,pseudo_fuc_ms

def pearson_divergence(a1,a2,a3,Xs1, Ys1, Xs2, Ys2, Xs3, Ys3, Xt, Yt, sigma, lamda):
    # 数据初始化
    ms1 = Xs1.shape[0]
    ms2 = Xs2.shape[0]
    ms3 = Xs3.shape[0]
    mt = Xt.shape[0]
    m = ms1+ms2+ms3+mt

    # 以下开始计算梯度
    # 计算特征x的核函数k(x,xj)
    data_X = anp.vstack((Xs1, Xs2, Xs3, Xt))
    data_Y = np.hstack((Ys1,Ys2,Ys3,Yt))

    data_X_norm = anp.sum(data_X ** 2, axis = -1)
    
    #=======================
    Xs1_W_norm = anp.sum(Xs1 ** 2, axis = -1)
    pair_dist1 = Xs1_W_norm[:, None] + data_X_norm[None, :] - 2 * anp.dot(Xs1, data_X.T)
    k_x1 = anp.exp(- pair_dist1 / sigma)

    l_y1 = anp.array(data_Y[:ms1, None] == data_Y, dtype=np.float64)
    phi_i1 = k_x1 * l_y1

    #=======================
    Xs2_W_norm = anp.sum(Xs2 ** 2, axis = -1)
    pair_dist2 = Xs2_W_norm[:, None] + data_X_norm[None, :] - 2 * anp.dot(Xs2, data_X.T)
    k_x2 = anp.exp(- pair_dist2 / sigma)

    l_y2 = anp.array(data_Y[ms1:ms1+ms2, None] == data_Y, dtype=np.float64)
    phi_i2 = k_x2 * l_y2
    #=======================
    Xs3_W_norm = anp.sum(Xs3 ** 2, axis = -1)
    pair_dist3 = Xs3_W_norm[:, None] + data_X_norm[None, :] - 2 * anp.dot(Xs3, data_X.T)
    k_x3 = anp.exp(- pair_dist3 / sigma)

    l_y3 = anp.array(data_Y[ms1+ms2:ms1+ms2+ms3, None] == data_Y, dtype=np.float64)
    phi_i3 = k_x3 * l_y3


    #=======================
    Xtk_W_norm = anp.sum(Xt ** 2, axis = -1)
    pair_dist4 = Xtk_W_norm[:, None] + data_X_norm[None, :] - 2 * anp.dot(Xt, data_X.T)
    k_x4 = anp.exp(- pair_dist4 / sigma)

    l_y4 = anp.array(data_Y[ms2+ms1+ms3:, None] == data_Y, dtype=np.float64)
    phi_i4 = k_x4 * l_y4
    

    b1 = anp.ones(ms1)
    #b1 = b1.T
    temp = anp.dot(phi_i1.T,b1)
    b1 = 1.0/ms1*temp 

    b2 = anp.ones(ms2)
    #b2 = b2[:, None]
    temp = anp.dot(phi_i2.T,b2)
    b2 = 1.0/ms2*temp
    
    b3 = anp.ones(ms3)
    #b2 = b2[:, None]
    temp = anp.dot(phi_i3.T,b3)
    b3 = 1.0/ms3*temp

    temp = anp.dot(phi_i4.T,phi_i4)
    H = 1.0 /mt*temp

    x1 = a1*b1 + a2*b2 + a3*b3
    I = anp.eye(m)

    theta = alina.solve(H + lamda*I, x1)
    D = 2 * anp.dot(x1.T,theta) - anp.dot(anp.dot(theta.T,H),theta) - 1
    #A = anp.linalg.inv(H + lamda*I)
    return D,I,H,b1,b2,b3


def pearson_divergence2(a1,a2,Xs1, Ys1, Xs2, Ys2, Xt, Yt, sigma, lamda):
    # 数据初始化
    ms1 = Xs1.shape[0]
    ms2 = Xs2.shape[0]
    mt = Xt.shape[0]
    m = ms1+ms2+mt

    # 以下开始计算梯度
    # 计算特征x的核函数k(x,xj)
    data_X = anp.vstack((Xs1, Xs2, Xt))
    data_Y = np.hstack((Ys1,Ys2,Yt))

    data_X_norm = anp.sum(data_X ** 2, axis = -1)
    Xs1_W_norm = anp.sum(Xs1 ** 2, axis = -1)
    pair_dist1 = Xs1_W_norm[:, None] + data_X_norm[None, :] - 2 * anp.dot(Xs1, data_X.T)
    k_x1 = anp.exp(- pair_dist1 / sigma)

    l_y1 = anp.array(data_Y[:ms1, None] == data_Y, dtype=np.float64)
    phi_i1 = k_x1 * l_y1


    Xs2_W_norm = anp.sum(Xs2 ** 2, axis = -1)
    pair_dist2 = Xs2_W_norm[:, None] + data_X_norm[None, :] - 2 * anp.dot(Xs2, data_X.T)
    k_x2 = anp.exp(- pair_dist2 / sigma)

    l_y2 = anp.array(data_Y[ms1:ms1+ms2, None] == data_Y, dtype=np.float64)
    phi_i2 = k_x2 * l_y2



    Xtk_W_norm = anp.sum(Xt ** 2, axis = -1)
    pair_dist4 = Xtk_W_norm[:, None] + data_X_norm[None, :] - 2 * anp.dot(Xt, data_X.T)
    k_x4 = anp.exp(- pair_dist4 / sigma)
    print('len::::::',ms2+ms1)

    l_y4 = anp.array(data_Y[ms2+ms1:, None] == data_Y, dtype=np.float64)
    phi_i4 = k_x4 * l_y4
    

    b1 = anp.ones(ms1)
    #b1 = b1.T
    temp = anp.dot(phi_i1.T,b1)
    b1 = 1.0/ms1*temp 

    b2 = anp.ones(ms2)
    #b2 = b2[:, None]
    temp = anp.dot(phi_i2.T,b2)
    b2 = 1.0/ms2*temp
    


    temp = anp.dot(phi_i4.T,phi_i4)
    H = 1.0 /mt*temp

    x1 = a1*b1 + a2*b2 
    I = anp.eye(m)

    theta = alina.solve(H + lamda*I, x1)
    D = 2 * anp.dot(x1.T,theta) - anp.dot(anp.dot(theta.T,H),theta) - 1
    #A = anp.linalg.inv(H + lamda*I)
    # print(b1,b2,phi_i4)
    return D

def pearson_divergence2_torch(w,data_X,data_Y, l, sigma=None,lamda=1e-2,device=torch.device('cpu')):
    # Xs1, Xs2, Xt,l=torch.tensor(Xs1),torch.tensor(Xs2),torch.tensor(Xt),torch.tensor(l)
    # 数据初始化
    unique_l,count_l=torch.unique(l,return_counts=True)
    ms1 = count_l[0]
    ms2 = count_l[1]
    mt = count_l[2]
    m = ms1+ms2+mt
    Xs1,Xs2,Xt=data_X[l==0],data_X[l==1],data_X[l==2]
    # 以下开始计算梯度
    # 计算特征x的核函数k(x,xj)
    data_X = torch.vstack((Xs1,Xs2,Xt))
    data_X_norm = torch.sum(data_X ** 2, axis = -1).to(device=device)
    
    X_sum=None
    for i in range(len(unique_l)-1):
        # print('{}/{}'.format(i,len(unique_l)-1))
        msi=count_l[i]
        Xsi=data_X[l==i]
        Xsi_W_norm = torch.sum(Xsi ** 2, axis = -1)
        pair_disti = Xsi_W_norm[:, None] + data_X_norm[None, :] - 2 * torch.matmul(Xsi, data_X.T)
        k_xi = torch.exp(- pair_disti / sigma)

        l_yi = torch.tensor(data_Y[sum(count_l[:i]):sum(count_l[:i])+msi, None] == data_Y, dtype=torch.double)
        phi_ii = k_xi * l_yi
        
        bi = torch.ones(msi).to(torch.double)
        phi_ii = phi_ii.to(torch.double)
        temp = torch.matmul(phi_ii.T,bi)
        bi = 1.0/msi*temp 
        
        if isinstance(X_sum,type(w[i]*bi)):
            X_sum+=w[i]*bi
        else:
            # print('initialise!!!')
            X_sum=w[i]*bi


    Xtk_W_norm = torch.sum(Xt ** 2, axis = -1)
    pair_dist4 = Xtk_W_norm[:, None] + data_X_norm[None, :] - 2 * torch.matmul(Xt, data_X.T)
    k_x4 = torch.exp(- pair_dist4 / sigma)

    l_y4 = torch.tensor(data_Y[sum(count_l[:-1]):, None] == data_Y, dtype=torch.double)
    phi_i4 = k_x4 * l_y4
    
    print('len::::::',sum(count_l[:-1]))
    temp = torch.matmul(phi_i4.T,phi_i4)
    H = 1.0 /mt*temp

    I = torch.eye(l.shape[0])

    theta = torch.linalg.solve(H + lamda*I, X_sum)
    D = 2 * torch.matmul(X_sum.T,theta) - torch.matmul(torch.matmul(theta.T,H),theta) - 1
    return D


def RMI(FX,y,l,alpha=0.5,sigma=None,lamda=1e-2,device=torch.device('cpu')):
    """
    
    """        

    if sigma is None:
        pairwise_dist = torch.cdist(FX,FX,p=2)**2 
        sigma = torch.median(pairwise_dist[pairwise_dist!=0])
        
    m = FX.shape[0]    
    Deltay = torch.as_tensor(y[:,None]==y,dtype=torch.float64,device=device)
    Deltal = torch.as_tensor(l[:,None]==l,dtype=torch.float64,device=device)
    FX_norm = torch.sum(FX ** 2, axis = -1)
    K = torch.exp(-(FX_norm[:,None] + FX_norm[None,:] - 2 * torch.matmul(FX, FX.t())) / sigma) * Deltay
                    
    P = K * Deltal
    H = ((1.0 - alpha) / m**2) * torch.matmul(K,K) * torch.matmul(Deltal,Deltal) + 1.0 * alpha / m * torch.matmul(P,P)
    h = torch.mean(P,axis=0)
    theta = torch.matmul(torch.inverse(H + lamda * torch.eye(m,device=device)),h)
    D = 2 * torch.matmul(h,theta) - torch.matmul(theta,torch.matmul(H,theta)) - 1
    return D


domain=['dslr.csv','webcam.csv','amazon.csv']
domain_num=3
# root=osp.join('Office-31_Alex','Data_office31')
root='resnet50Data'
dataSet='Office31'

scs=domain[:2]
tg=domain[2]
Cs_end,Cu_start=10,21

# make_print_to_file(filename='{}_Mlti_2{}_v1_2'.format(dataSet,tg),path=dataset+'logs')
print("DataSet:{}".format(root),"\nTarget Domain:{}".format(tg))
Xs,ys,Xt,yt,l,Cs,Cu=roadDataMSbyCSV(root,scs,tg,Cs_end,Cu_start)
Xt_new,Yt_new,Yt_pseudo,Yt_conf=pseudo_fuc_ms(Xs,ys,Xt,yt,Cs,Cu,conf=0.0)

l=torch.Tensor(l)
data_X = torch.tensor(np.vstack((Xs[0],Xs[1],Xt)))
data_Y =torch.tensor(np.hstack((ys[0],ys[1],yt)))

for a in range(1,11):
    a_1=a/10
    b_1=1-a_1
    w=[a_1,b_1]
    pearson_div=pearson_divergence2(a_1,b_1,Xs[0],ys[0],Xs[1],ys[1],Xt,yt,1,1)
    pearson_div_torch=pearson_divergence2_torch(w,data_X,data_Y,l,1,1)

    print(pearson_div)
    print(pearson_div_torch)
    break