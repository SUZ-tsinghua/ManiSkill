import csv
import os
import matplotlib.pyplot as plt
import numpy as np

file_names = os.listdir("./results")

for filename in file_names:
    if filename.endswith(".csv"):
        steps = []
        mean = []
        std = []
        with open(f"./results/{filename}") as csvfile:
            csv_reader = csv.reader(csvfile)
            header = next(csv_reader)
            for row in csv_reader:
                steps.append(float(row[0]))
                mean.append(float(row[4]))
                std.append(float(row[5]))

        steps = np.array(steps)
        mean = np.array(mean)
        std = np.array(std)

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

        plt.plot(steps, mean, color="blue")

        plt.fill_between(steps, mean - std, mean + std, color="blue", alpha=0.2)

        plt.xlabel("Steps")
        plt.ylabel("Success Rate (Once)")
        output_filename = filename.replace("csv", "png")
        plt.savefig(f"./results/{output_filename}")
        # plt.show()