import numpy as np
def compute_acc(Y, Y_pseudo, Cs):
    acc_known = 0
    acc_unknown = 0
    known_i = 0
    unknown_i = 0
    mtk = Y[Y < Cs].shape
    mtu = Y[Y >= Cs].shape
    mt = Y.shape
    # 计算已知类的分类准确率
    for c in range(Cs):  
        known_i += np.sum(Y_pseudo[Y == c] == Y[Y == c]) 
    acc_known = known_i/mtk
    # 计算未知类的分类准确率
    unknown_i = np.sum(Y_pseudo[Y >= Cs ] == Cs)
    acc_unknown = unknown_i/mtu
    # 计算总体分类准确率
    acc = (unknown_i+known_i)/mt
    return acc,acc_unknown,acc_known