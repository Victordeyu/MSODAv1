import matplotlib.pyplot as plt
import numpy as np

# 2D# 数据
os_star_a = [98.0,98.6,99.3,99.3,99.3,99.3,99.3]
os_a = [93.78,96.12,96.87,96.87,96.87,96.87,96.87]
unk_a = [51.5,70.7,72.3,72.3,72.3,72.3,72.3]

# 数据
d_values = [20, 50, 100, 150, 200, 250, 300]
data = {
    20: (np.array([0.85540353]), np.array([0.89064744]), np.array([0.50296443]), np.array([0.64288198])),
    50: (np.array([0.86208376]), np.array([0.89345024]), np.array([0.54841897]), np.array([0.67965258])),
    100: (np.array([0.84831642]), np.array([0.87998601]), np.array([0.53162055]), np.array([0.66281733])),
    150: (np.array([0.84831642]), np.array([0.87998601]), np.array([0.53162055]), np.array([0.66281733])),
    200: (np.array([0.84831642]), np.array([0.87998601]), np.array([0.53162055]), np.array([0.66281733])),
    250: (np.array([0.84831642]), np.array([0.87998601]), np.array([0.53162055]), np.array([0.66281733])),
    300: (np.array([0.84831642]), np.array([0.87998601]), np.array([0.53162055]), np.array([0.66281733]))
}

# 提取数据
os_values = [data[d][0][0] for d in d_values]
os_star_values = [data[d][1][0] for d in d_values]
unk_values = [data[d][2][0] for d in d_values]

# 转换为百分比
os_b = [val * 100 for val in os_values]
os_star_b = [val * 100 for val in os_star_values]
unk_b = [val * 100 for val in unk_values]

# # 创建图和子图
# fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# # 图 (a) A -> D
# axes[0].plot(d_values, os_star_a, marker='^', color='orange', label='OS*')
# axes[0].plot(d_values, os_a, marker='o', color='red', label='OS')
# axes[0].plot(d_values, unk_a, marker='s', color='blue', label='Unk') # 增加unk
# axes[0].set_xlabel('d')
# axes[0].set_ylabel('Accuracy(%)')
# axes[0].set_ylim(0, 100)
# axes[0].legend()

# # 图 (b) P -> I
# axes[1].plot(d_values, os_star_b, marker='^', color='orange', label='OS*')
# axes[1].plot(d_values, os_b, marker='o', color='red', label='OS')
# axes[1].plot(d_values, unk_b, marker='s', color='blue', label='Unk')  # 增加unk
# axes[1].set_xlabel('d')
# axes[1].set_ylabel('Accuracy(%)')
# axes[1].set_ylim(0, 100)
# axes[1].legend()



# # 调整布局，避免重叠
# plt.tight_layout()

# # 显示图表
# plt.show()

# # 保存图表
# plt.savefig('PCA_2DA_line_chart.png')


def drawDataLine(x, os,os_star,unk, x_label,pic_name,values):
    fig, axes = plt.subplots(1, 1)
    axes.plot(x, os, marker='o', color='red', label='OS')
    axes.plot(x, os_star, marker='^', color='orange', label='OS*')
    axes.plot(x, unk, marker='s', color='blue', label='Unk')  # 增加unk
    axes.set_xlabel(x_label)
    axes.set_ylabel('Accuracy(%)')
    axes.set_ylim(0, 100)
    axes.legend()    
    # axes.set_xticks(x) # 注意这里也要用x
    # axes.set_xticklabels(values) # 使用给定的 values 作为刻度标签
    
    plt.tight_layout()
    plt.savefig(pic_name)
    plt.show()

# 示例调用
drawDataLine(d_values,os_a,os_star_a,unk_a,"",'sl_D_PCA.png',d_values)
drawDataLine(d_values,os_b,os_star_b,unk_b,"",'sl_A_PCA.png',d_values)