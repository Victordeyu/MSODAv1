import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def process_csv_and_draw(filename, key,sample_interval=2):
    # 读取CSV文件
    df = pd.read_csv(filename)
    
    x_data = df[key].tolist()
    
    # 计算需要绘制的数据
    acc_data = [
        (df['acc_tar'] * 100).tolist(), 
        (df['acc_k_tar'] * 100).tolist(), 
        (df['acc_un_tar'] * 100).tolist(),
        (df['HOS'] ).tolist(),  # 归一化到百分比
    ]
    
    loss_data = [
        df['avg_cls_loss'].tolist(),
        df['avg_div_loss'].tolist(),
        df['avg_open_loss'].tolist(),
        df['avg_total_loss'].tolist()
    ]
    x_data = x_data[::sample_interval]
    acc_data = [data[::sample_interval] for data in acc_data]
    loss_data = [data[::sample_interval] for data in loss_data]
    
    # 调用绘图函数
    drawDataLine(x_data, acc_data, loss_data, key)
    
def drawDataLine(x_data, acc_data, loss_data, x_label):
    fig, ax1 = plt.subplots(figsize=(12, 5))  # 创建主坐标轴
    # ax2 = ax1.twinx()  # 创建共享 x 轴的次坐标轴

    # 生成均匀分布的 X 轴数据
    x = np.arange(len(x_data))

    # # 画前四个指标 (Accuracy 类别)
    # ax1.plot(x, acc_data[0], marker='o', color='red', label='Acc Tar')
    # ax1.plot(x, acc_data[1], marker='^', color='orange', label='Acc K Tar')
    # ax1.plot(x, acc_data[2], marker='s', color='blue', label='Acc Un Tar')
    # ax1.plot(x, acc_data[3], marker='x', color='black', label='HOS')

    # 画后四个指标 (Loss 类别)
    ax1.plot(x, loss_data[0], marker='o', linestyle='dashed', color='green', label='Cls Loss')
    ax1.plot(x, loss_data[1], marker='^', linestyle='dashed', color='purple', label='Div Loss')
    ax1.plot(x, loss_data[2], marker='s', linestyle='dashed', color='brown', label='Open Loss')
    ax1.plot(x, loss_data[3], marker='x', linestyle='dashed', color='cyan', label='Total Loss')

    # 设置 X 轴
    ax1.set_xlabel(x_label)
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_data)

    # 设置 Y 轴
    # ax1.set_ylabel('Accuracy (%)', color='black')
    # ax1.set_ylim(0, 100)  # 确保 Accuracy 轴在 0-100 之间
    ax1.set_ylabel('Loss', color='black')

    # 添加图例
    # ax1.legend(loc="upper left")  # Accuracy 图例
    ax1.legend(loc="upper right") # Loss 图例

    # plt.title(f"{x_label} vs Accuracy & Loss")
    # plt.grid()
    plt.tight_layout()
    
    # 保存和显示
    plt.savefig('dl_2WconveyLoss.png')
    plt.show()

# 示例调用
process_csv_and_draw("2WconveyLoss.csv", "epoch")

