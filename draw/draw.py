import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['Webcam-OS*','Dslr-OS*','Amazon-OS*']
# jaoa_sl = [97.37, 100, 96.87,99.33,84.83,87.99]
# jaoa_sl_wo_progress = [91.05, 88.39,100,97.00,80.4,83.8]
jaoa_sl = [100,99.33,87.99]
jaoa_sl_wo_progress = [ 88.39,97.00,83.8]

x = np.arange(len(labels))  # x 轴的位置
width = 0.3   # 柱子的宽度

# 绘制柱状图
fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, jaoa_sl, width, label='Pro', color='#1f77b4') # 蓝色
rects2 = ax.bar(x + width/2, jaoa_sl_wo_progress, width, label='No Pro', color='#ff7f0e') # 橘色

# 添加标签和标题
ax.set_ylabel('Accuracy(%)')
#ax.set_title('Title of the Chart') # 可以添加标题
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylim([60, 100]) # 设定y轴范围
ax.legend()

# # 将图例放在图表外部
# ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

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
plt.savefig('ablu-o31-osStar.png')  # 可以指定文件名和格式，例如 '柱状图.pdf'

# 显示图表
plt.show()