import numpy as np
from scipy.stats import wilcoxon
from scipy.stats import ttest_ind

# 生成两个配对样本
before = np.array([85, 90, 78, 92, 88, 76, 94, 89, 75, 86])
after = np.array([88, 85, 80, 90, 84, 78, 96, 87, 74, 83])


# HyMOS:{o3 wda ,home,DN}
HyMOS={
    'OS':np.array([96.1, 96.7, 49.6, 69.5, 52.5, 50.1, 71.5, 43.6, 47.8]),
    'OS*':np.array([96.6, 97.3, 48.0, 69.4, 51.7, 49.4, 71.5, 43.2, 47.4]),
    'Unk':np.array([84.6, 83.6, 83.1, 72.7,86.0, 84.1, 70.6, 86, 85.5]),
    'HOS':np.array([90.2, 89.9, 60.8, 71.0,64.6, 62.2, 71.1, 57.5, 61.00])
}
DL={
    'OS':np.array([92.3, 95.6, 62.0,69.8, 60.1, 55.9, 70.4, 56.62, 60.96]),
    'OS*':np.array([91.9, 95.5, 61.1, 69.9, 59.8, 55.5, 70.3, 56.48, 60.91]),
    'Unk':np.array([98.5, 97.9, 79.4, 79.1, 73.0, 74.9, 73.6, 70.74, 66.12]),
    'HOS':np.array([95.1, 96.6, 69.1,74.0, 65.8, 63.8, 71.9, 62.81, 63.41])
}


# 进行Wilcoxon符号秩检验
for key in HyMOS.keys():
    stat, p = wilcoxon(HyMOS[key], DL[key])

    print(f"{key} Wilcoxon 统计量: {stat}, p值: {p}")

    # 解释结果
    alpha = 0.05
    if p < alpha:
        print("拒绝原假设，样本之间存在显著差异")
    else:
        print("无法拒绝原假设，样本之间无显著差异")


for key in HyMOS.keys():
    stat, p = ttest_ind(DL[key],HyMOS[key])

    print(f"{key} t-test 统计量: {stat}, p值: {p}")

    # 解释结果
    alpha = 0.1
    if p < alpha:
        print("拒绝原假设，样本之间存在显著差异")
    else:
        print("无法拒绝原假设，样本之间无显著差异")