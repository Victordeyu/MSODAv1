import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.linalg as la
from sklearn.svm import SVC
# from JDIP import JDIP
# from MSJDIP import MSJDIP
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import scale,LabelEncoder
from os import path as osp
import autograd.numpy as anp
import autograd.numpy.linalg as alina
import pymanopt as pm
import pymanopt.manifolds as manifolds
import pymanopt.optimizers as optimizers
from sklearn.metrics import pairwise_distances
import autograd.numpy as anp
import autograd.numpy.linalg as alina



def readDataMS(root,scs,tg,Cs_end,Cu_start,fn='fea',postfix=''): 
    '''
    root: 数据集路径,eg 'OSDA-2/Office-31_Alex/Data_office31'
    scs: List of source domain name, eg ['dslr','webcam']
    tg: name of target domain name, eg 'amazon'
    Cs_end: Known end
    Cu_start: Unknown start
    fn: 在mat文件中feature列名 
    postfix: 每个数据集文件的后缀名

    return: 
        Xs: List of numpy array
        Xt: Numpy array
        l: numpy array of label of the sample

    '''
    Xs,ys,l=[],[],[]
    li=0
    for sc in scs:
        data = sio.loadmat(osp.join(root ,sc + postfix+'.mat'))# source domain 
        Xsi,ysi = data[fn].astype(np.float64),data['labels'].ravel()
        ysi = LabelEncoder().fit(ysi).transform(ysi).astype(np.float64)
        Xsi = Xsi / la.norm(Xsi,axis=1,keepdims=True)
        Xsn=Xsi[ysi[:]<Cs_end,:]
        ysn=ysi[ysi[:]<Cs_end]#筛选出已知类
        Xs.append(Xsn)
        ys.append(ysn)
        l=np.hstack((np.array(l),np.full((Xsn.shape[0],),li)))
        li+=1

    data = sio.loadmat(osp.join(root , tg + postfix+'.mat'))# target domain 
    Xt,yt = data[fn].astype(np.float64),data['labels'].ravel()
    yt = LabelEncoder().fit(yt).transform(yt).astype(np.float64)
    C=len(np.unique(yt))
    # index_unknwon=yt[yt[:]>Cs_end and yt[:]<Cu_start]
    Xt=np.vstack((Xt[yt[:]<Cs_end,:],Xt[yt[:]>Cu_start,:]))#筛选已知类和未知类
    yt=np.hstack((yt[yt[:]<Cs_end],yt[yt[:]>Cu_start]))

    l=np.hstack((np.array(l),np.full((Xt.shape[0],),li)))
    
    Xt = Xt / la.norm(Xt,axis=1,keepdims=True)

    Cs = Cs_end
    Cu = C - Cu_start

    return Xs,ys,Xt,yt,l,Cs,Cu

def compute_one_unknown(Ytu, Ytu_pseudo, Cs):
    """
    :param Ytu: 实际未知类的标签,全部为未知类
    :param Ytu_pseudo: 实际未知类的伪标签,可能混合已知类
    :param Cs:
    :return:
    """
    # 计算预测未知类中真实未知类的准确率
    one_unknown = Ytu_pseudo[Ytu_pseudo >= Cs].shape[0] / Ytu.shape[0]

    return one_unknown

def compute_acc_known(Ytk, Ytk_pseudo, Cs):
    """
    :param Ytk: 实际已知类的标签
    :param Ytk_pseudo: 实际已知类的伪标签
    :param Cs:
    :return:
    """
    # 计算已知类的平均准确率
    acc_known = 0
    for c in range(Cs):  # 计算已知类的分类准确率
        known_i = np.sum(Ytk_pseudo[Ytk == c] == Ytk[Ytk == c]) / Ytk[Ytk == c].shape[0]
        acc_known = acc_known + known_i

    acc_known = acc_known / Cs

    return acc_known

def cla_Svc(Xs, Xt, Ys):
    # 2.SVM分类模型
    model_cla = LinearSVC(dual=False)  # dual决定无法收敛时取消默认1000的迭代次数
    model_cla.fit(Xs, Ys)
    conf_matrix = model_cla.decision_function(Xt)
    conf_label = conf_matrix.argmax(axis=1)  # 每个样本最大置信度的索引,即类
    conf_vec = np.max(conf_matrix, axis=1)  # 每个样本最大置信度

    return conf_label, conf_vec

def pseudo_fuc_ms(Xs, ys, Xt, Yt,Cs,Cu):
    # 1.加载数据
    Xs=np.vstack(Xs)
    ys=np.hstack(ys)

    Xs = scale(Xs, with_std=False)
    Xt = scale(Xt, with_std=False)


    # 2.设置分类模型,标注已知类伪标签
    Xt_label, Xt_labelConf = cla_Svc(Xs, Xt, ys)

    conf = 0  # 根据SVM定义,置信度大于0才属于已知类
    Xtk = Xt[Xt_labelConf >= conf]
    Ytk = Yt[Xt_labelConf >= conf]
    Ytk_pseudo = Xt_label[Xt_labelConf >= conf]
    Xtu = Xt[Xt_labelConf < conf]
    Ytu = Yt[Xt_labelConf < conf]
    Ytu_pseudo = Xt_label[Xt_labelConf < conf]
    Ytu_pseudo[:] = Cs
    print("pseudo Xtk:", compute_acc_known(Ytk, Ytk_pseudo, Cs))
    print("pseudo Xtu:", compute_one_unknown(Ytu, Ytu, Cs))

    # 5.目标域数据整合
    Xt_new = np.vstack((Xtk, Xtu))
    Yt_new = np.hstack((Ytk, Ytu))
    Yt_pseudo = np.hstack((Ytk_pseudo, Ytu_pseudo))

    # 计算已知类准确率,后续应删
    Ytk_new = Yt_new[Yt_new < Cs]
    Ytk_new_pseudo = Yt_pseudo[Yt_new < Cs]
    acc_known = compute_acc_known(Ytk_new, Ytk_new_pseudo, Cs)
    print("predict Xtk :", acc_known)
    # 计算未知类准确率,后续应删
    Ytu_new = Yt_new[Yt_new >= Cs]
    Ytu_new_pseudo = Yt_pseudo[Yt_new >= Cs]
    one_unknown = compute_one_unknown(Ytu_new, Ytu_new_pseudo, Cs)
    print("predict Xtu one :", one_unknown)

    # 计算总体准确率
    one_acc = (acc_known * Cs + one_unknown) / (Cs + 1)
    print("predict Xt one :", one_acc)

    return Yt_pseudo


def cost_manifold(Ws,Wt,l, Cs, Xs, Ys ,Xt,Yt, Maxiter, dimension, truncated_param=1e-8):
    # FX=np.vstack(FX,Xt)
    # y=np.vstack(ys,Yt_pseudo)
    # print(Ws,Wt,l,Cs,Xs)
    dim = Xs.shape[1]
    manifold = manifolds.Product([manifolds.Stiefel(dim, dimension), manifolds.Stiefel(dim, dimension)])

    # 2.切分Xt已知类和未知类
    Xtk = Xt[Yt < Cs, :]
    Ytk = Yt[Yt < Cs]
    lk=l[np.hstack((Ys,Yt))<Cs]
    # 融合源域和目标域
    Yk_all=np.hstack((Ys,Ytk))
    Xk_all=np.vstack((Xs,Xtk))

    @pm.function.autograd(manifold)
    def cost(Ws,Wt):
        # 1.更新Xs
        Xs_W = anp.dot(Xs, Ws)
        Xtk_W = anp.dot(Xtk, Wt)
        # print(lk,Ytk)
        X_all_W=anp.vstack((Xs_W,Xtk_W))
    
        # pairwise_dist = torch.cdist(torch.Tensor(Xk_all).to(torch.float32),torch.tensor(Xk_all).to(torch.float32),p=2)**2 
        # sigma = np.float(torch.median(pairwise_dist[pairwise_dist!=0]))
        # print(sigma)

        # # print(type(Xall_W[0]),type(Xall_W))
        known_dist = pairwise_distances(Xk_all,Xk_all, 'euclidean')**2
        known_sigma = np.median(known_dist[known_dist != 0])
        # # print("sigma",known_sigma)

        known_rcs=RMI_np(X_all_W,Yk_all,lk,sigma=known_sigma)
        # 没有*2开根号和外部参数
        # print('type of rcs:',type(known_rcs),known_rcs)
        if known_rcs<0:
            known_rcs=0
        # if known_div > 0:
        #     div = anp.sqrt(2 * known_div)
        # else:
        #     div = 0
        return known_rcs

    problem = pm.Problem(manifold=manifold, cost=cost)
    optimizer = optimizers.SteepestDescent(max_iterations=Maxiter)
    if Ws is None and Wt is None:
        W = optimizer.run(problem)
    else:
        W = optimizer.run(problem, initial_point=[Ws[:, :dimension], Wt[:, :dimension]])
    return W

def RMI_np(FX,y,l,sigma=0.8,alpha=0.5,lamda=1e-2):
    '''
    FX: numpy array of feature. eg: vstack(Xs)
    y: numpy array of labels
    l: numpy array of domain labels
    
    return -->float RCS Divergence of FX,y,l
    '''
    # print(FX)
    m=FX.shape[0]
    # print('m:',m)
    
    Deltay=(y[:,None]==y).astype(float)
    Deltal=(l[:,None]==l).astype(float)
    FX_norm=anp.sum(FX**2,axis=-1)#这里由于做了归一化 所以始终都等于1 后面要考虑下要不要改
    # print(FX_norm,FX_norm.shape)
    # print('1',FX_norm[:,None],FX_norm[None,:])
    # print(FX_norm[:,None].shape,FX_norm[None,:].shape)
    A=-(FX_norm[:,None] + FX_norm[None,:])
    # print("A",A.shape)
    B=2 * anp.dot(FX, FX.T)
    # print("B",B.shape)
    K=anp.exp(-(FX_norm[:,None] + FX_norm[None,:] - 2 * anp.dot(FX, FX.T)) / sigma) * Deltay
    # print('K is',K.shape)
    P = K * Deltal
    # print("P is",P.shape)
    H = ((1.0 - alpha) / m**2) * anp.dot(K,K) * anp.dot(Deltal,Deltal) + 1.0 * alpha / m * anp.dot(P,P)
    h = anp.mean(P,axis=0)
    h=h.reshape((h.shape[0],))#对齐一下向量

    # theta = anp.matrix(H + lamda * anp.eye(m)).I.dot(h)
    theta=alina.solve(H + lamda * anp.eye(m),h)
    
    D = 2 * anp.dot(h.T,theta)-anp.dot(theta.T,anp.dot(H,theta)) - 1
    # print(D,type(D))

    # print(H.shape,h.shape,theta.shape)
    return D


def model_loss(Xs, Ys, Xt, Yt_pseudo, Cs, sigma, eta, gamma_tk, gamma_tu, gamma_s):
    # 数据初始化
    ms = Xs.shape[0]
    mt = Xt.shape[0]
    data_X = anp.vstack((Xs, Xt))

    # 初始化源域已知类的标签矩阵
    Y0 = np.zeros([Cs + 1, ms + mt])
    for index in range(ms):
        Y0[Ys[index], index] = 1

    # 初始化被视为未知类的标签矩阵
    Yu = np.zeros([Cs + 1, ms + mt])
    Yu[Cs, :] = 1

    # 初始化W,V1,V2矩阵
    W = np.zeros([ms + mt, ms + mt])
    w1, w2 = np.diag_indices_from(W)
    W[w1[:ms], w2[:ms]] = np.sqrt(1 / ms)
    mtk = Yt_pseudo[Yt_pseudo < Cs].shape[0]
    mtu = Yt_pseudo[Yt_pseudo >= Cs].shape[0]
    V1 = np.zeros([ms + mt, ms + mt])
    v11, v21 = np.diag_indices_from(V1)
    V1[v11[ms:][Yt_pseudo < Cs], v21[ms:][Yt_pseudo < Cs]] = np.sqrt(1 / mtk)
    V2 = np.zeros([ms + mt, ms + mt])
    v12, v22 = np.diag_indices_from(V2)
    V2[v12[ms:][Yt_pseudo >= Cs], v22[ms:][Yt_pseudo >= Cs]] = np.sqrt(1 / mtu)

    # 计算特征x的核矩阵K
    data_norm = np.sum(data_X ** 2, axis=-1)
    pair_dist = data_norm[:, None] + data_norm[None, :] - 2 * np.dot(data_X, data_X.T)
    K = np.exp(- pair_dist / sigma)

    # 计算alpha
    V_sub_W = gamma_tk * np.dot(V1, V1) + gamma_tu * np.dot(V2, V2) - gamma_s * np.dot(W, W)
    alpha_l = np.dot(np.dot(W, W), K) + np.dot(V_sub_W, K) + eta * np.eye(ms + mt)
    alpha_r = np.dot(np.dot(W, W), Y0.T) + np.dot(V_sub_W, Yu.T)
    alpha = nlina.solve(alpha_l, alpha_r)
    predict_alpha = np.dot(alpha.T, K)

    # 计算源域损失
    known_fro = nlina.norm(np.dot(Y0 - predict_alpha, W), ord='fro')
    known_loss = known_fro * known_fro

    # 计算被视为未知类损失
    target_tk_fro = nlina.norm(np.dot(Yu - predict_alpha, V1), ord='fro')
    unknown_target_tk = gamma_tk * target_tk_fro * target_tk_fro
    target_tu_fro = nlina.norm(np.dot(Yu - predict_alpha, V2), ord='fro')
    unknown_target_tu = gamma_tu * target_tu_fro * target_tu_fro
    source_fro = nlina.norm(np.dot(Yu - predict_alpha, W), ord='fro')
    unknown_source = gamma_s * source_fro * source_fro
    unknown_loss = unknown_target_tk + unknown_target_tu - unknown_source

    # 计算正则化项
    regular_norm = eta * np.sum(np.dot(predict_alpha, alpha).diagonal())

    loss = known_loss + unknown_loss + regular_norm
    print('loss is', format(loss, '.2f'), '| known_loss is', format(known_loss, '.2f'), '| unknown_loss is',
          format(unknown_loss, '.2f'), '(', format(unknown_target_tk, '.2f'), '+', format(unknown_target_tu, '.2f'), '-', format(unknown_source, '.2f'), ')')
    return loss, predict_alpha


scs=['dslr','webcam']
tg='amazon'
root=osp.join('Office-31_Alex','Data_office31')

Xs,ys,Xt,yt,l,Cs,Cu=readDataMS(root,scs,tg,9,20,fn='fts',postfix='_Al7')

Yt_pseudo=pseudo_fuc_ms(Xs,ys,Xt,yt,Cs,Cu)

Xs_all=np.vstack(Xs)
ys_all=np.hstack(ys)

np.random.seed(0)  # 使得PCA算法稳定
pca = PCA(svd_solver="full").fit(np.vstack((Xs_all,Xt)))
W0 = pca.components_.T  # 初始化降维矩阵为Dxd维,D＞d
Ws, Wt = W0, W0

W = cost_manifold(Ws, Wt,l, Cs, Xs_all, ys_all, Xt, Yt_pseudo, 50, 100).point
# print(W)
Xs_pre = np.dot(Xs_all, W[0])
Xt_pre = np.dot(Xt, W[1])

distance = pairwise_distances(Xs_pre, Xt_pre, 'euclidean')
sigma = np.median(distance[distance != 0])
loss, predict = model_loss(Xs_pre, ys , Xt_pre, Yt_pseudo, Cs, sigma, eta=0.001, gamma_tk=0.2, gamma_tu=0.7, gamma_s=0.7)

predict_Xt = predict[:, Xs_pre.shape[0]:]  # 分类模型对目标域样本的置信度矩阵
predict_Xt = predict_Xt / np.sum(predict_Xt, axis=0)[None, :]  # 置信度归1

Yt_pseudo = predict_Xt.argmax(axis=0)

acc_known = compute_acc_known(yt[yt < Cs], Yt_pseudo[yt < Cs], Cs)
one_unknown = compute_one_unknown(yt[yt >= Cs], Yt_pseudo[yt >= Cs], Cs)
one_acc = (acc_known * Cs + one_unknown) / (Cs + 1)
print('Xtk is', format(acc_known * 100, '.1f'), '| Xtu is', format(one_unknown * 100, '.1f'), '| Xt is',
              format(one_acc * 100, '.1f'), '| Hos is',
              format(2 * acc_known * one_unknown * 100 / (acc_known + one_unknown), '.1f'))
