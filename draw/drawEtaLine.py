import matplotlib.pyplot as plt
import numpy as np

# 数据
eta_values =  [0.005, 0.001, 0.0005, 0.0001]

data_a = {
    0.005 :(np.array([0.91389426]), np.array([0.98666667]), np.array([0.18617021]), np.array([0.31323698])),
    0.001 :(np.array([0.96879433]), np.array([0.99333333]), np.array([0.72340426]), np.array([0.83714781])),
    0.0005: (np.array([0.98401032]), np.array([0.98666667]), np.array([0.95744681]), np.array([0.97183715])),
    0.0001: (np.array([0.36038129]), np.array([0.29641941]), np.array([1.]), np.array([0.45728938]))
}

data_b = {
    0.005 :(np.array([0.80775992]), np.array([0.87509718]), np.array([0.13438735]), np.array([0.23299414])),
    0.001 :(np.array([0.84831642]), np.array([0.87998601]), np.array([0.53162055]), np.array([0.66281733])),
    0.0005: (np.array([0.76963136]), np.array([0.77030991]), np.array([0.76284585]), np.array([0.76655971])),
    0.0001: (np.array([0.41059347]), np.array([0.3522457]), np.array([0.99407115]), np.array([0.5201707]))
}

def extract_data(data,y_values):
    # 提取数据
    os_values = [data[d][0][0] for d in y_values]
    os_star_values = [data[d][1][0] for d in y_values]
    unk_values = [data[d][2][0] for d in y_values]

    # 转换为百分比
    os = [val * 100 for val in os_values]
    os_star  = [val * 100 for val in os_star_values]
    unk  = [val * 100 for val in unk_values]
    
    return os ,os_star,unk

os_a,os_star_a,unk_a=extract_data(data_a,eta_values)
os_b,os_star_b,unk_b=extract_data(data_b,eta_values)

# 创建图和子图
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 生成均匀分布的X轴数据
x = np.arange(len(eta_values))

# # 图 (a) A -> D
# axes[0].plot(x, os_star_a, marker='^', color='orange', label='OS*')
# axes[0].plot(x, os_a, marker='o', color='red', label='OS')
# axes[0].plot(x, unk_a, marker='s', color='blue', label='Unk') # 增加unk
# axes[0].set_xlabel(r'$\eta$')
# axes[0].set_ylabel('Accuracy(%)')
# axes[0].set_ylim(0, 100)
# axes[0].legend()
# # 设置自定义 x 轴刻度
# axes[0].set_xticks(x)
# axes[0].set_xticklabels(eta_values)

# # 图 (b) P -> I
# axes[1].plot(x, os_star_b, marker='^', color='orange', label='OS*')
# axes[1].plot(x, os_b, marker='o', color='red', label='OS')
# axes[1].plot(x, unk_b, marker='s', color='blue', label='Unk')  # 增加unk
# axes[1].set_xlabel(r'$\eta$')
# axes[1].set_ylabel('Accuracy(%)')
# axes[1].set_ylim(0, 100)
# axes[1].legend()

# axes[1].set_xticks(x) # 注意这里也要用x
# axes[1].set_xticklabels(eta_values) # 使用给定的 eta_values 作为刻度标签

def drawDataLine(x, os,os_star,unk, x_label,pic_name,values):
    fig, axes = plt.subplots(1, 1)
    axes.plot(x, os, marker='o', color='red', label='OS')
    axes.plot(x, os_star, marker='^', color='orange', label='OS*')
    axes.plot(x, unk, marker='s', color='blue', label='Unk')  # 增加unk
    axes.set_xlabel(x_label)
    axes.set_ylabel('Accuracy(%)')
    axes.set_ylim(0, 100)
    axes.legend()    
    axes.set_xticks(x) # 注意这里也要用x
    axes.set_xticklabels(values) # 使用给定的 values 作为刻度标签
    
    plt.tight_layout()
    plt.savefig(pic_name)
    plt.show()

# 示例调用
drawDataLine(x,os_a,os_star_a,unk_a,"",'sl_D_eta.png',eta_values)
drawDataLine(x,os_b,os_star_b,unk_b,"",'sl_A_eta.png',eta_values)