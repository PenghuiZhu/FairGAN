import pandas as pd
import matplotlib.pyplot as plt

df1 = pd.read_csv("ctgan_test.csv")
df2 = pd.read_csv("fairgan_test.csv")

# 提取性别数据
sex_data1 = df1['sex'].value_counts(normalize=True)
sex_data2 = df2['sex'].value_counts(normalize=True)

# 绘制饼状图
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# CTGAN数据的饼状图
axs[0].pie(sex_data1, labels=sex_data1.index, autopct='%1.1f%%', startangle=90)
axs[0].set_title('CTGAN Test Data Sex Distribution')

# FairGAN数据的饼状图
axs[1].pie(sex_data2, labels=sex_data2.index, autopct='%1.1f%%', startangle=90)
axs[1].set_title('FairGAN Test Data Sex Distribution')

plt.tight_layout()
plt.show()