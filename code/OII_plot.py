import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams['font.family'] = ['Times New Roman', 'serif']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 0.8
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

df_100 = pd.read_csv('results_child_100_results.csv')
df_1000 = pd.read_csv('results_child_1000_results.csv')

fig, ax = plt.subplots(1, 1, figsize=(7.2, 4.8))  


colors = ['#1f77b4', '#ff7f0e']  
alpha_fill = 0.2
alpha_line = 0.8

def compress_x_axis(year):
    if year <= 1500:
        # 350-1500年压缩到0-0.4的范围 (4/10)
        return (year - 350) / (1500 - 350) * 0.4
    else:
        # 1501-2050年映射到0.4-1.0的范围 (6/10)
        return 0.4 + (year - 1501) / (2050 - 1501) * 0.6


# 应用变换到数据
df_100['Year_compressed'] = df_100['Year'].apply(compress_x_axis)
df_1000['Year_compressed'] = df_1000['Year'].apply(compress_x_axis)

# 绘制100样本大小的数据
ax.plot(df_100['Year_compressed'], df_100['Mean_OII'],
        color=colors[0], linewidth=1.5, alpha=alpha_line,
        label='Sample size = 100', zorder=3)
ax.fill_between(df_100['Year_compressed'], df_100['CI_Lower'], df_100['CI_Upper'],
                color=colors[0], alpha=alpha_fill, zorder=1)

# 绘制1000样本大小的数据
ax.plot(df_1000['Year_compressed'], df_1000['Mean_OII'],
        color=colors[1], linewidth=1.5, alpha=alpha_line,
        label='Sample size = 1,000', zorder=3)
ax.fill_between(df_1000['Year_compressed'], df_1000['CI_Lower'], df_1000['CI_Upper'],
                color=colors[1], alpha=alpha_fill, zorder=1)

# 设置坐标轴
ax.set_xlabel('Year', fontsize=11, fontweight='bold')
ax.set_ylabel('Occupational Inheritance Index (OII)', fontsize=11, fontweight='bold')

# 设置图例
ax.legend(loc='upper left', frameon=True, fancybox=False,
          shadow=False, fontsize=10, framealpha=1.0)

# 设置网格
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# 设置历史时期的起止年份
historical_periods = {
    'Post-Classical\n(501–1500)': 501,
    'Early Modern\n(1501–1750)': 1501,
    'Mid-Modern\n(1751–1900)': 1751,
    'Contemporary\n(1901–present)': 1901,
}

# 设置自定义x轴刻度
x_ticks_original = [350, 1000, 1500, 1600, 1700, 1800, 1900, 2000]
x_ticks_compressed = [compress_x_axis(year) for year in x_ticks_original]
ax.set_xticks(x_ticks_compressed)
ax.set_xticklabels(x_ticks_original)
ax.set_xlim(0, 1)

# 添加历史时期标注
for period_name, start_year in historical_periods.items():
    if start_year >= 350 and start_year <= 2050:
        # 添加垂直线标记时期开始
        compressed_year = compress_x_axis(start_year)
        ax.axvline(x=compressed_year, color='gray', linestyle='--', alpha=0.5, linewidth=0.8)

        # 添加时期标签在x轴下方，垂直排列
        ax.text(compressed_year, ax.get_ylim()[0] - (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.12, period_name,
                rotation=90, fontsize=8, ha='center', va='top')

# 设置y轴范围，留出一些边距
y_min = min(df_100['CI_Lower'].min(), df_1000['CI_Lower'].min())
y_max = max(df_100['CI_Upper'].max(), df_1000['CI_Upper'].max())
y_margin = (y_max - y_min) * 0.05
ax.set_ylim(y_min - y_margin, y_max + y_margin)

# 调整布局
plt.tight_layout()

# 保存图片
plt.savefig('professional_oii_trends.png', dpi=300, bbox_inches='tight')
plt.savefig('professional_oii_trends.pdf', bbox_inches='tight')

print("专业图表已生成：")
print("- professional_oii_trends.png (高分辨率PNG)")
print("- professional_oii_trends.pdf (矢量PDF)")

# 显示图表
plt.show()
