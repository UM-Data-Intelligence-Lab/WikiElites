import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('filtered_living_data.csv')

# 数据预处理：将出生和死亡年份转换为数值型（float），无法解析则转为 NaN
df['Node1_Birth'] = pd.to_numeric(df['Node1_Birth'], errors='coerce')
df['Node2_Birth'] = pd.to_numeric(df['Node2_Birth'], errors='coerce')
df['Node1_Death'] = pd.to_numeric(df['Node1_Death'], errors='coerce')
df['Node2_Death'] = pd.to_numeric(df['Node2_Death'], errors='coerce')

# 合并 Node1 和 Node2 的个体信息
df_all = pd.concat([
    df[['Node1', 'Node1_Birth', 'Node1_Death', 'Node1_occ_level1']].rename(
        columns={'Node1': 'Individual', 'Node1_Birth': 'Birth', 'Node1_Death': 'Death', 'Node1_occ_level1': 'Occupation'}),
    df[['Node2', 'Node2_Birth', 'Node2_Death', 'Node2_occ_level1']].rename(
        columns={'Node2': 'Individual', 'Node2_Birth': 'Birth', 'Node2_Death': 'Death', 'Node2_occ_level1': 'Occupation'})
])

# 去重个体
df_all.drop_duplicates(subset='Individual', inplace=True)

# 定义时间轴范围
years = range(-500, 2020)
occupations = df_all['Occupation'].dropna().unique()

# 初始化 DataFrame 存储每年各领域人数
data = {occupation: [0] * len(years) for occupation in occupations}
df_years = pd.DataFrame(data, index=years)

# 计算每年各领域人数
for _, row in df_all.iterrows():
    birth = row['Birth']
    death = row['Death']
    occupation = row['Occupation']

    # 检查是否为空
    if pd.isna(birth) or pd.isna(occupation):
        continue  # 跳过无出生年或职业的数据

    birth = int(birth)
    death = int(death) if not pd.isna(death) else 2025  # 如果没有死亡年，默认为 2025

    # 确保年份在有效范围内
    start_year = max(birth, min(years))
    end_year = min(death, max(years))

    if start_year > end_year:
        continue  # 跳过无效的时间段

    for year in range(start_year, end_year + 1):
        df_years.at[year, occupation] += 1

# 计算总人数及各领域的占比
df_years['Total'] = df_years.sum(axis=1)
for occupation in occupations:
    df_years[occupation] = df_years[occupation] / df_years['Total']

# 绘制堆叠面积图
fig, ax = plt.subplots(figsize=(14, 8))
df_years[occupations].plot.area(ax=ax, stacked=True, alpha=0.6)

ax.set_xlabel('Year')
ax.set_ylabel('Share of Individuals')
ax.set_title('Share of Individuals by Domain of Influence')
ax.set_xlim([-500, 2019])
ax.set_xticks([-500, 50, 1500, 1750, 1900, 2019])
ax.set_xticklabels([
    'Ancient History\nBefore 500AD',
    'Post-Classical History\n501-1500AD',
    'Early Modern Period\n1501-1750AD',
    'Mid Modern Period\n1751-1900AD',
    'Late Modern Period\n1901-1950',
    'Contemporary Era\n1951-2019'
])

plt.tight_layout()
plt.show()

# 保存处理后的数据
df_years.to_csv('processed_data.csv', index=True)
