import matplotlib.pyplot as plt
import numpy as np

# 数据
values =  [i/10 for i in range(1,10)]

# D
data_a = {
    0.1: (np.array([0.96395874]), np.array([0.99333333]), np.array([0.67021277]), np.array([0.80039222])),
    0.2: (np.array([0.96879433]), np.array([0.99333333]), np.array([0.72340426]), np.array([0.83714781])),
    0.3: (np.array([0.97869117]), np.array([0.98666667]), np.array([0.89893617]), np.array([0.94076052])),
    0.4: (np.array([0.98159252]), np.array([0.98666667]), np.array([0.93085106]), np.array([0.95794652])),
    0.5: (np.array([0.96079948]), np.array([0.96166667]), np.array([0.95212766]), np.array([0.95687339])),
    0.6: (np.array([0.96031593]), np.array([0.96166667]), np.array([0.94680851]), np.array([0.95417975])),
    0.7: (np.array([0.92673506]), np.array([0.91994048]), np.array([0.99468085]), np.array([0.95585186])),
    0.8: (np.array([0.83102031]), np.array([0.81625]), np.array([0.9787234]), np.array([0.8901335])),
    0.9: (np.array([0.62441548]), np.array([0.60068681]), np.array([0.86170213]), np.array([0.70790074])),
    1.0: (np.array([0.35862329]), np.array([0.32054945]), np.array([0.7393617]), np.array([0.44721105]))
}

data_b = {
    0.1: (np.array([0.94922631]), np.array([1.]), np.array([0.44148936]), np.array([0.61254613])),
    0.2: (np.array([0.96879433]), np.array([0.99333333]), np.array([0.72340426]), np.array([0.83714781])),
    0.3: (np.array([0.97578981]), np.array([0.98666667]), np.array([0.86702128]), np.array([0.92298274])),
    0.4: (np.array([0.97601547]), np.array([0.98]), np.array([0.93617021]), np.array([0.95758383])),
    0.5: (np.array([0.97746615]), np.array([0.98]), np.array([0.95212766]), np.array([0.96586279])),
    0.6: (np.array([0.97843327]), np.array([0.98]), np.array([0.96276596]), np.array([0.97130654])),
    0.7: (np.array([0.97285622]), np.array([0.97333333]), np.array([0.96808511]), np.array([0.97070213])),
    0.8: (np.array([0.96776273]), np.array([0.96666667]), np.array([0.9787234]), np.array([0.97265767])),
    0.9: (np.array([0.94415584]), np.array([0.93857143]), np.array([1.]), np.array([0.96831245])),
    1.0: (np.array([0.94415584]), np.array([0.93857143]), np.array([1.]), np.array([0.96831245]))
}

data_c = {
    0.1: (np.array([0.97575758]), np.array([0.97333333]), np.array([1.00000000]), np.array([0.98648649])),
    0.2: (np.array([0.98449387]), np.array([0.98666667]), np.array([0.96276596]), np.array([0.97456980])),
    0.3: (np.array([0.98255964]), np.array([0.98666667]), np.array([0.94148936]), np.array([0.96354875])),
    0.4: (np.array([0.96879433]), np.array([0.99333333]), np.array([0.72340426]), np.array([0.83714781])),
    0.5: (np.array([0.96615087]), np.array([1.00000000]), np.array([0.62765957]), np.array([0.77124183])),
    0.6: (np.array([0.95116054]), np.array([1.00000000]), np.array([0.46276596]), np.array([0.63272727])),
    0.7: (np.array([0.93230174]), np.array([1.00000000]), np.array([0.25531915]), np.array([0.40677966])),
    0.8: (np.array([0.91924565]), np.array([1.00000000]), np.array([0.11170213]), np.array([0.20095694])),
    0.9: (np.array([0.91150870]), np.array([1.00000000]), np.array([0.02659574]), np.array([0.05181347])),
}
# data_a={
# 0.1: (np.array([0.83489857]), np.array([0.87224218]), np.array([0.46146245]), np.array([0.60359244])),
# 0.2: (np.array([0.84831642]), np.array([0.87998601]), np.array([0.53162055]), np.array([0.66281733])),
# 0.3: (np.array([0.79810062]), np.array([0.82296998]), np.array([0.54940711]), np.array([0.65892321])),
# 0.4: (np.array([0.83665039]), np.array([0.85015732]), np.array([0.70158103]), np.array([0.76875621])),
# 0.5: (np.array([0.74121638]), np.array([0.74043683]), np.array([0.74901186]), np.array([0.74469966])),
# 0.6: (np.array([0.69401679]), np.array([0.68268725]), np.array([0.80731225]), np.array([0.73978787])),
# 0.7: (np.array([0.66221004]), np.array([0.64572354]), np.array([0.8270751]), np.array([0.72523405])),
# 0.8: (np.array([0.69059032]), np.array([0.66488651]), np.array([0.94762846]), np.array([0.78146918])),
# 0.9: (np.array([0.62703588]), np.array([0.59863275]), np.array([0.91106719]), np.array([0.72252061])),
# 1.0: (np.array([0.39036257]), np.array([0.33710634]), np.array([0.9229249]), np.array([0.49383511])),
# }
# data_b={
#     0.1: (np.array([0.85016297]), np.array([0.90148362]), np.array([0.33695652]), np.array([0.49055384])),
#     0.2: (np.array([0.84831642]), np.array([0.87998601]), np.array([0.53162055]), np.array([0.66281733])),
#     0.3: (np.array([0.79033943]), np.array([0.80178444]), np.array([0.67588933]), np.array([0.73347387])),
#     0.4: (np.array([0.77652139]), np.array([0.77660436]), np.array([0.7756917]), np.array([0.77614776])),
#     0.5: (np.array([0.76448121]), np.array([0.75486214]), np.array([0.86067194]), np.array([0.80430202])),
#     0.6: (np.array([0.74237265]), np.array([0.72925813]), np.array([0.87351779]), np.array([0.79489583])),
#     0.7: (np.array([0.74760795]), np.array([0.7282976]), np.array([0.94071146]), np.array([0.82098763])),
#     0.8: (np.array([0.73281055]), np.array([0.71122994]), np.array([0.9486166]), np.array([0.81294808])),
#     0.9: (np.array([0.72845909]), np.array([0.70565283]), np.array([0.95652174]), np.array([0.81215569])),
#     1.0: (np.array([0.72400801]), np.array([0.6997685]), np.array([0.96640316]), np.array([0.81175128]))
# }
# data_c={
#     0.1: (np.array([0.71999566]), np.array([0.70197546]), np.array([0.90019763]), np.array([0.78882444])),
#     0.2: (np.array([0.77431021]), np.array([0.77348036]), np.array([0.7826087]), np.array([0.77801775])),
#     0.3: (np.array([0.8244896]), np.array([0.83549587]), np.array([0.71442688]), np.array([0.77023285])),
#     0.4: (np.array([0.84831642]), np.array([0.87998601]), np.array([0.53162055]), np.array([0.66281733])),
#     0.5: (np.array([0.85253635]), np.array([0.89737496]), np.array([0.4041502]), np.array([0.55730658])),
#     0.6: (np.array([0.83398719]), np.array([0.88922385]), np.array([0.28162055]), np.array([0.427766])),
#     0.7: (np.array([0.82211128]), np.array([0.88663466]), np.array([0.17687747]), np.array([0.29492037])),
#     0.8: (np.array([0.78758815]), np.array([0.85923234]), np.array([0.07114625]), np.array([0.13141135])),
#     0.9: (np.array([0.79687891]), np.array([0.87429407]), np.array([0.02272727]), np.array([0.04430289])),
#     1.0: (np.array([0.79463061]), np.array([0.87389604]), np.array([0.00197628]), np.array([0.00394365]))
# }

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
os_b,os_star_b,unk_b=extract_data(data_b,values)
os_c,os_star_c,unk_c=extract_data(data_c,values)

# # 创建图和子图
# fig, axes = plt.subplots(1, 3, figsize=(12, 5))

# # 生成均匀分布的X轴数据
x = np.arange(len(values))

# # 图 (a) A -> D
# axes[0].plot(x, os_star_a, marker='^', color='orange', label='OS*')
# axes[0].plot(x, os_a, marker='o', color='red', label='OS')
# axes[0].plot(x, unk_a, marker='s', color='blue', label='Unk') # 增加unk
# axes[0].set_xlabel('Gamma_tk')
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
# axes[1].set_xlabel('tu')
# axes[1].set_ylabel('Accuracy(%)')
# axes[1].set_ylim(0, 100)
# axes[1].legend()

# axes[1].set_xticks(x) # 注意这里也要用x
# axes[1].set_xticklabels(values) # 使用给定的 values 作为刻度标签

# # 图 (b) P -> I
# axes[2].plot(x, os_star_c, marker='^', color='orange', label='OS*')
# axes[2].plot(x, os_c, marker='o', color='red', label='OS')
# axes[2].plot(x, unk_c, marker='s', color='blue', label='Unk')  # 增加unk
# axes[2].set_xlabel('ys')
# axes[2].set_ylabel('Accuracy(%)')
# axes[2].set_ylim(0, 100)
# axes[2].legend()

# axes[2].set_xticks(x) # 注意这里也要用x
# axes[2].set_xticklabels(values) # 使用给定的 values 作为刻度标签



# # 调整布局，避免重叠
# plt.tight_layout()

# # 显示图表
# plt.show()

# # 保存图表
# plt.savefig('ytku_2A_line_chart.png')

def drawDataLine(x, os,os_star,unk, x_label,pic_name):
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
drawDataLine(x,os_a,os_star_a,unk_a,"",'sl_D_tk.png')
drawDataLine(x,os_b,os_star_b,unk_b,"",'sl_D_tu.png')
drawDataLine(x,os_c,os_star_c,unk_c,"",'sl_D_ys.png')
