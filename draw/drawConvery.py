import matplotlib.pyplot as plt
import numpy as np

# 数据
values =  [i for i in range(0,10)]

data_a = {
    0: (np.array([0.79407005]), np.array([0.79650077]), np.array([0.76976285])),
    1: (np.array([0.82521781]), np.array([0.83204789]), np.array([0.75691700])),
    2: (np.array([0.84180100]), np.array([0.85295738]), np.array([0.73023715])),
    3: (np.array([0.84209195]), np.array([0.85515490]), np.array([0.71146245])),
    4: (np.array([0.83901249]), np.array([0.85413903]), np.array([0.68774704])),
    5: (np.array([0.84085078]), np.array([0.86238646]), np.array([0.62549407])),
    6: (np.array([0.84393072]), np.array([0.86864000]), np.array([0.59683794])),
    7: (np.array([0.84733504]), np.array([0.87791835]), np.array([0.54150198])),
    8: (np.array([0.85394261]), np.array([0.88597718]), np.array([0.53359684])),
    9: (np.array([0.84831642]), np.array([0.87998601]), np.array([0.53162055])),
}

data_d = {
    0: (np.array([0.97917473]), np.array([0.98666667]), np.array([0.90425532])),
    1: (np.array([0.97337202]), np.array([0.98666667]), np.array([0.84042553])),
    2: (np.array([0.97604771]), np.array([0.99333333]), np.array([0.80319149])),
    3: (np.array([0.98065764]), np.array([1.00000000]), np.array([0.78723404])),
    4: (np.array([0.97920696]), np.array([1.00000000]), np.array([0.77127660])),
    5: (np.array([0.97775629]), np.array([1.00000000]), np.array([0.75531915])),
    6: (np.array([0.97485493]), np.array([1.00000000]), np.array([0.72340426])),
    7: (np.array([0.96879433]), np.array([0.99333333]), np.array([0.72340426])),
    8: (np.array([0.96879433]), np.array([0.99333333]), np.array([0.72340426])),
    9: (np.array([0.96879433]), np.array([0.99333333]), np.array([0.72340426])),
}

data_w = {
    0: (np.array([0.96988369]), np.array([0.98709677]), np.array([0.79775281])),
    1: (np.array([0.96504003]), np.array([0.98064516]), np.array([0.80898876])),
    2: (np.array([0.96469955]), np.array([0.98064516]), np.array([0.80524345])),
    3: (np.array([0.97111382]), np.array([0.99032258]), np.array([0.77902622])),
    4: (np.array([0.96838995]), np.array([0.99032258]), np.array([0.74906367])),
    5: (np.array([0.97391457]), np.array([0.99677419]), np.array([0.74531835])),
    6: (np.array([0.97289312]), np.array([0.99677419]), np.array([0.73408240])),
    7: (np.array([0.97187167]), np.array([0.99677419]), np.array([0.72284644])),
    8: (np.array([0.97153119]), np.array([0.99677419]), np.array([0.71910112])),
    9: (np.array([0.97378277]), np.array([1.00000000]), np.array([0.71161049])),
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

os_a,os_star_a,unk_a=extract_data(data_a,values)
os_b,os_star_b,unk_b=extract_data(data_d,values)
os_c,os_star_c,unk_c=extract_data(data_w,values)

# 创建图和子图
# fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# 生成均匀分布的X轴数据
x = np.arange(len(values))

# 图 (a) A -> D
# axes[0].plot(x, os_star_a, marker='^', color='orange', label='OS*')
# axes[0].plot(x, os_a, marker='o', color='red', label='OS')
# axes[0].plot(x, unk_a, marker='s', color='blue', label='Unk') # 增加unk
# axes[0].set_xlabel('Iteration')
# axes[0].set_ylabel('Accuracy(%)')
# axes[0].set_ylim(0, 100)
# axes[0].legend()
# # 设置自定义 x 轴刻度
# axes[0].set_xticks(x)
# axes[0].set_xticklabels(values)

# # 图 (b) P -> I
# axes[1].plot(x, os_star_b, marker='^', color='orange', label='OS*')
# axes[1].plot(x, os_b, marker='o', color='red', label='OS')
# axes[1].plot(x, unk_b, marker='s', color='blue', label='Unk')  # 增加unk
# axes[1].set_xlabel('D')
# axes[1].set_ylabel('Accuracy(%)')
# axes[1].set_ylim(0, 100)
# axes[1].legend()

# axes[1].set_xticks(x) # 注意这里也要用x
# axes[1].set_xticklabels(values) # 使用给定的 values 作为刻度标签

# # 图 (b) P -> I
# axes[2].plot(x, os_star_c, marker='^', color='orange', label='OS*')
# axes[2].plot(x, os_c, marker='o', color='red', label='OS')
# axes[2].plot(x, unk_c, marker='s', color='blue', label='Unk')  # 增加unk
# axes[2].set_xlabel('W')
# axes[2].set_ylabel('Accuracy(%)')
# axes[2].set_ylim(0, 100)
# axes[2].legend()

# axes[2].set_xticks(x) # 注意这里也要用x
# axes[2].set_xticklabels(values) # 使用给定的 values 作为刻度标签

def drawDataLine(x, os,os_star,unk, x_label):
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
    plt.savefig('sl_W_convey.png')
    plt.show()

# 示例调用
# drawDataLine(x,os_a,os_star_a,unk_a,"Iteration")
# drawDataLine(x,os_b,os_star_b,unk_b,"Iteration")
drawDataLine(x,os_c,os_star_c,unk_c,"Iteration")