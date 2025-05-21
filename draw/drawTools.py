import matplotlib.pyplot as plt
import numpy as np


def drawDataLine(x_data,y_data,x_label):
    data_num=len(y_data)
    
    fig, axes = plt.subplots(1,data_num, figsize=(12, 5))

    # 生成均匀分布的X轴数据
    x = np.arange(len(x_data))

    for i in range(data_num):
    # 图 (a) A -> D\
        print(y_data[0][0])
        axes[i].plot(x, y_data[i][0], marker='o', color='red', label='OS')
        axes[i].plot(x, y_data[i][1], marker='^', color='orange', label='OS*')
        axes[i].plot(x, y_data[i][2], marker='s', color='blue', label='Unk') # 增加unk
        axes[i].set_xlabel(x_label[i])
        axes[i].set_ylabel('Accuracy(%)')
        axes[i].set_ylim(0, 100)
        axes[i].legend()
        # 设置自定义 x 轴刻度
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(x_data)
    
    # 调整布局，避免重叠
    plt.tight_layout()

    # 显示图表
    plt.show()

    # 保存图表
    plt.savefig('dl_ent_2Every_line_chart.png')

x_data = [0.5,0.7,0.9,1.0,1.1,1.3,1.5]

#webcam
y_data_1={
    0.5:(0.9064,0.9041,0.9513,0.92713),
    0.7:(0.9456,0.9476,0.9064,0.926529),
    0.9:(0.864,0.8597,0.9513,0.903165),
    1.0:(0.9079,0.9057,0.9513,0.927943),
    1.1:(0.9066,0.9043,0.9513,0.927218),
    1.3:(0.9613,0.9635,0.9176,0.939999),
    1.5:(0.9505,0.9525,0.9101,0.930835)
}

#dslr
y_data_2={
    0.5:(0.9578,0.9676,0.7606,0.85174,),
    0.7:(0.8721,0.8663,0.9894,0.923722,),
    0.9:(0.9444,0.9432,0.9681,0.955494,),
    1.0:(0.894,0.8921,0.9309,0.911075,),
    1.1:(0.9599,0.9611,0.9362,0.948485,),
    1.3:(0.9423,0.9472,0.8457,0.893586,),
    1.5:(0.9576,0.9592,0.9255,0.942088,)
}

#amazon
y_data_3={
    0.5:(0.2079,0.1689,0.9872,0.288459,),
    0.7:(0.3454,0.3142,0.9704,0.474692,),
    0.9:(0.6613,0.6531,0.8261,0.729467,),
    1.0:(0.4832,0.4592,0.9634,0.621923,),
    1.1:(0.2982,0.2639,0.9842,0.41621,),
    1.3:(0.3497,0.3177,0.9881,0.480862,),
    1.5:(0.1915,0.1522,0.9763,0.263387,)
}
def extract_data(y_data,x_data):
    # 提取数据
    os_values = [y_data[d][0] for d in x_data]
    os_star_values = [y_data[d][1] for d in x_data]
    unk_values = [y_data[d][2] for d in x_data]
    HOS= [y_data[d][3] for d in x_data]

    # 转换为百分比
    os = [val * 100 for val in os_values]
    os_star  = [val * 100 for val in os_star_values]
    unk  = [val * 100 for val in unk_values]
    HOS  = [val * 100 for val in unk_values]
    
    return [os ,os_star,unk,HOS]
# print(y_data[0.5])
y_data_1=extract_data(y_data_1,x_data)
y_data_2=extract_data(y_data_2,x_data)
y_data_3=extract_data(y_data_3,x_data)

# drawDataLine(x_data,[y_data_1,y_data_2,y_data_3],["->W Ent","->D Ent","->A Ent"])
drawDataLine(x_data,[y_data_1,y_data_2],["->W Ent","->D Ent"])
