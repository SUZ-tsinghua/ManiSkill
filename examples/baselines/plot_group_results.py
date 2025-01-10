import csv
import os
import matplotlib.pyplot as plt
import numpy as np

base_dir = "./results/roll_ball"
# 定义每个文件后缀对应的颜色和标签
file_configs = {
    'rgb.csv': {'color': 'blue', 'label': 'RGB'},
    'adapt.csv': {'color': 'green', 'label': 'Adapt'},
    'state.csv': {'color': 'red', 'label': 'State'},
}

# 初始化图形
plt.figure(figsize=(12, 8))
plt.clf()

# 设置字体大小
plt.rcParams.update({
    'font.size': 14,        # 全局字体大小
    'axes.titlesize': 18,   # 标题字体大小
    'axes.labelsize': 16,   # 坐标轴标签字体大小
    'legend.fontsize': 14,  # 图例字体大小
    'xtick.labelsize': 14,  # x轴刻度字体大小
    'ytick.labelsize': 14   # y轴刻度字体大小
})

# 遍历每个文件配置
for suffix, config in file_configs.items():
    # 在base_dir目录下查找匹配的文件
    matching_files = [f for f in os.listdir(base_dir) if f.endswith(suffix)]
    if not matching_files:
        print(f"未找到以 {suffix} 结尾的文件，跳过。")
        continue
    elif len(matching_files) > 1:
        print(f"找到多个以 {suffix} 结尾的文件，使用第一个文件: {matching_files[0]}")

    filename = matching_files[0]
    steps = []
    mean = []
    std = []

    # 读取CSV文件内容
    csv_path = os.path.join(base_dir, filename)
    with open(csv_path, 'r') as csvfile:
        csv_reader = csv.reader(csvfile)
        header = next(csv_reader)  # 跳过标题行
        for row in csv_reader:
            try:
                steps.append(float(row[0]))
                mean.append(float(row[4]))
                std.append(float(row[5]))
            except ValueError as e:
                print(f"在文件 {filename} 中解析行时出错: {row}，错误: {e}")

    # 将列表转换为NumPy数组
    steps = np.array(steps)#[::4]
    mean = np.array(mean)#[::4]
    std = np.array(std)#[::4]

    # 绘制均值曲线
    plt.plot(steps, mean, color=config['color'], label=config['label'])

    # 绘制均值±标准差的填充区域
    plt.fill_between(steps, mean - std, mean + std, color=config['color'], alpha=0.2)

# 添加坐标轴标签
plt.xlabel("Steps")
plt.ylabel("Success Rate (Once)")

# 添加图例
plt.legend()

# 添加标题（可选）
plt.title("Comparison of Success Rates for RGB, Adapt, and State Methods")

# 保存合并后的图像
output_filename = "combined_results.png"
output_path = os.path.join(base_dir, output_filename)
plt.savefig(output_path)
print(f"图像已保存到 {output_path}")