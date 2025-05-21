import matplotlib.pyplot as plt
import numpy as np

# 数据 sampler的数据
# labels = ['Sketch-OS','Clipart-OS','Avg-OS']
# labels = ['Sketch-OS*','Clipart-OS*','Avg-OS*']
# labels = ['Sketch-UNK','Clipart-UNK','Avg-UNK']
# labels = ['Sketch-HOS','Clipart-HOS','Avg-HOS']
# os_dl = [56.62,60.96,58.79 ]
# os_dl_wo_progress = [50.26,51.41,50.84 ]
# os_star_dl = [56.48,60.91,58.70]
# os_star_dl_wo_progress = [50.00,51.14,50.57]
# unk_dl = [70.74,66.12,68.43 ]
# unk_dl_wo_progress = [75.92,78.59,77.26]
# hos_dl = [62.81,63.41,63.11]
# hos_dl_wo_progress = [60.29,61.96,61.13 ]

# Pro的数据
# labels = ['Sketch-UNK','Clipart-UNK','Avg-UNK']
# unk_dl = [70.74,66.12,68.43]
# unk_dl_wo_progress = [75.95,63.05,69.50  ]
# labels = ['Sketch-OS*','Clipart-OS*','Avg-OS*']
# os_star_dl = [56.48,60.91,58.70]
# os_star_dl_wo_progress = [53.16,62.61,57.89]
# os_dl = [56.62,60.96,58.79]
# os_dl_wo_progress = [53.39,62.61,58.00]
# hos_dl = [62.81,63.41,63.11]
# hos_dl_wo_progress = [62.54,62.83,62.69 ]


# Pro2的数据
# labels = ['W-OS','D-OS','A-OS','Avg-OS']
# labels = ['W-OS*','D-OS*','A-OS*','Avg-OS*']
labels = ['W-UNK','D-UNK','A-UNK','Avg-UNK']
# labels = ['W-HOS','D-HOS','A-HOS','Avg-HOS']
unk_dl = [88.76,93.62,81.72,88.03]
unk_dl_wo_progress = [98.80,99.47,79.70,92.66 ]
# labels = ['Sketch-OS*','Clipart-OS*','Avg-OS*']
# os_star_dl = [95.30,95.62,73.26,88.06 ]
# os_star_dl_wo_progress = [77.70,82.35,65.20,75.08 ]
# os_dl = [94.99,95.53,73.66,88.06 ]
# os_dl_wo_progress = [78.70,83.16,65.20,75.69 ]
# hos_dl = [91.91,94.60,77.25,87.92]
# hos_dl_wo_progress = [87.40,90.10,72.30,83.27]

x = np.arange(len(labels))  # x 轴的位置
width = 0.3   # 柱子的宽度

# 绘制柱状图
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, unk_dl, width, label='Pro', color='#1f77b4') # 蓝色
rects2 = ax.bar(x + width/2, unk_dl_wo_progress, width, label='No Pro', color='#ff7f0e') # 橘色

# 添加标签和标题
ax.set_ylabel('Accuracy(%)')
#ax.set_title('Title of the Chart') # 可以添加标题
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([0, 100]) # 设定y轴范围
ax.legend()

# 将图例放在图表外部
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# 在柱子上添加数值标签
def autolabel(rects):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


autolabel(rects1)
autolabel(rects2)

fig.tight_layout() # 调整布局，为图例留出空间

# 保存图表
plt.savefig('ablu/dl-ablu-pro-unk.png')  # 可以指定文件名和格式，例如 '柱状图.pdf'

# 显示图表
plt.show()