
import pandas as pd

# Step 1: 
df = pd.read_csv('Q_R_Q_living.csv')

# Step 2: 提取 Node1 和 Node2 的出生/死亡信息，并统一格式
node1_df = df[['Node1', 'Node1_Birth', 'Node1_Death']].copy()
node1_df.columns = ['node', 'birth', 'death']

node2_df = df[['Node2', 'Node2_Birth', 'Node2_Death']].copy()
node2_df.columns = ['node', 'birth', 'death']

# 合并两个节点信息，去重
combined_df = pd.concat([node1_df, node2_df], ignore_index=True)
combined_df = combined_df.drop_duplicates(subset=['node']).reset_index(drop=True)

# Step 3: 删除出生和死亡都缺失的行
combined_df = combined_df.dropna(subset=['birth', 'death'], how='all')

# Step 4: 确保 birth 和 death 是整数类型
combined_df['birth'] = combined_df['birth'].astype(int)
combined_df['death'] = combined_df['death'].astype(int)

# Step 5: 创建年份范围和计数 DataFrame
start_year = 300
end_year = 2020
years = list(range(start_year, end_year + 1))
year_counts = pd.DataFrame({'Year': years, 'Count': 0})

# Step 6: 对每个人物，更新对应年份的存活人数
for _, row in combined_df.iterrows():
    birth = row['birth']
    death = row['death']

    # 筛选年份范围内的有效区间
    valid_years = year_counts[(year_counts['Year'] >= birth) & (year_counts['Year'] <= death)]
    year_counts.loc[valid_years.index, 'Count'] += 1

# Step 7: 输出结果
print(year_counts.head())
print(year_counts.tail())

# 可选：保存为 CSV
year_counts.to_csv('living_300_to_2020.csv', index=False)

import pandas as pd
import matplotlib.pyplot as plt

# Step 1: 加载数据
year_counts = pd.read_csv('living_300_to_2020.csv')

# Step 2: 绘图
plt.figure(figsize=(14, 6))
plt.plot(year_counts['Year'], year_counts['Count'], color='blue', linewidth=1.5)

# 添加标题和标签
plt.title('Number of Living Individuals Over Time (300–2020 AD)', fontsize=14)
plt.xlabel('Year', fontsize=12)
plt.ylabel('Number of Living Individuals', fontsize=12)

# 设置网格
plt.grid(True, linestyle='--', alpha=0.5)

# 可选：标注历史时期
historical_periods = {
    'Post-Classical\n(501–1500)': (501, 1500),
    'Early Modern\n(1501–1750)': (1501, 1750),
    'Mid Modern\n(1751–1900)': (1751, 1900),
    'Contemporary\n(1901–2020)': (1901, 2020)
}

colors = ['#FFDDC1', '#C1E1FF', '#FFC1C1', '#D4FFC1']

for i, ((name, (start, end))) in enumerate(historical_periods.items()):
    plt.axvspan(start, end, color=colors[i], alpha=0.4)
    mid_year = (start + end) / 2
    plt.text(mid_year, max(year_counts['Count']) * 0.95, name,
             ha='center', va='top', fontsize=10, rotation=0)

# 设置坐标轴范围
plt.xlim(300, 2020)
plt.tight_layout()

# 显示图像
plt.show()
