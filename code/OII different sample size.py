import pandas as pd
import numpy as np
from collections import Counter
import math
import matplotlib.pyplot as plt
import os


def load_data(file_path):
    df = pd.read_csv(file_path)

    df = df.dropna(subset=['Node1_Birth', 'Node2_Birth', 'Node1_Death', 'Node2_Death',
                           'Node1_occ_level3', 'Node2_occ_level3'])

    df['Node1_Birth'] = df['Node1_Birth'].astype(int)
    df['Node2_Birth'] = df['Node2_Birth'].astype(int)
    df['Node1_Death'] = df['Node1_Death'].fillna(2025).astype(int)
    df['Node2_Death'] = df['Node2_Death'].fillna(2025).astype(int)
    return df


def calculate_oii(sample_df):
    parents = sample_df['Node1_occ_level3'].tolist()
    children = sample_df['Node2_occ_level3'].tolist()

    pair_counts = Counter()
    parent_counts = Counter()
    child_counts = Counter()

    for p, c in zip(parents, children):
        if p == 'Missing' or c == 'Missing':
            continue
        pair_counts[(p, c)] += 1
        parent_counts[p] += 1
        child_counts[c] += 1

    T = sum(pair_counts.values())
    if T == 0:
        return None

    oii = 0.0
    for (p, c), count in pair_counts.items():
        freq_p = parent_counts.get(p, 0)
        freq_c = child_counts.get(c, 0)

        if freq_p == 0 or freq_c == 0:
            continue

        try:
            pmi = math.log((count * T) / (freq_p * freq_c))
        except ValueError:
            pmi = float('-inf')

        weight = count / T
        oii += pmi * weight

    return oii


# ==================  Bootstrap ==================
def process_year(df_relation, year, sample_size=None, num_bootstrap_samples=500):
    df_year = df_relation[
        ((df_relation['Node1_Birth'] <= year) & (df_relation['Node1_Death'] > year)) &
        ((df_relation['Node2_Birth'] <= year) & (df_relation['Node2_Death'] > year))
        ]

    all_people = set(df_year['Node1']).union(set(df_year['Node2']))
    total_people = len(all_people)

    child_people = 0
    if 'Relation' in df_year.columns and 'child' in df_year['Relation'].unique():
        child_df = df_year[df_year['Relation'] == 'child']
        child_people = len(set(child_df['Node1']).union(set(child_df['Node2'])))

    oii_values = []
    if sample_size and len(df_year) >= sample_size:
        for _ in range(num_bootstrap_samples):
            sampled_df = df_year.sample(n=sample_size, replace=False)
            oii = calculate_oii(sampled_df)
            if oii is not None:
                oii_values.append(oii)
    else:
        oii = calculate_oii(df_year)
        if oii is not None:
            oii_values.append(oii)
    return oii_values, total_people, child_people


# ================= ==================
def save_results_and_plot(results_by_sample_size_dict, filename_prefix):
    fig, ax1 = plt.subplots(figsize=(14, 6))

    colors = ['blue', 'green', 'red']
    labels = ['Sample Size 100', 'Sample Size 1000', 'Full Data']

    people_data = []

    for idx, (sample_size, results_by_year) in enumerate(results_by_sample_size_dict.items()):
        mean_ci = {}
        people_yearly = {}

        for year, (oii_values, total_people, child_people) in results_by_year.items():
            if len(oii_values) >= 1 and oii_values[0] is not None:
                mean = np.mean(oii_values)
                ci_lower = np.percentile(oii_values, 2.5)
                ci_upper = np.percentile(oii_values, 97.5)
                mean_ci[year] = (mean, ci_lower, ci_upper)
                people_yearly[year] = (total_people, child_people)

        years_plot = list(mean_ci.keys())
        means_plot = [mean_ci[y][0] for y in years_plot]
        ci_lowers_plot = [mean_ci[y][1] for y in years_plot]
        ci_uppers_plot = [mean_ci[y][2] for y in years_plot]

        ax1.errorbar(years_plot, means_plot,
                     yerr=[np.array(means_plot) - np.array(ci_lowers_plot),
                           np.array(ci_uppers_plot) - np.array(means_plot)],
                     fmt='-o', capsize=5, ecolor=colors[idx], linestyle='-', marker='o', markersize=3, alpha=0.8,
                     label=labels[idx])

        people_data.append((years_plot, [p[0] for p in people_yearly.values()], [p[1] for p in people_yearly.values()]))

    ax1.set_xlabel('Year')
    ax1.set_ylabel('OII (with negative PMI)')
    ax1.set_title(f'OII Over Time ({filename_prefix})')
    ax1.grid(True)
    ax1.legend(loc='upper left')

    ax2 = ax1.twinx()
    for i, (years, total_people, child_people) in enumerate(people_data):
        if i == 2:
            ax2.plot(years, total_people, color='gray', linestyle='--', linewidth=1, label='Total People')
            ax2.plot(years, child_people, color='orange', linestyle='--', linewidth=1, label='Child Relation People')
    ax2.set_ylabel('Number of Living Individuals')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{filename_prefix}_plot.png')
    plt.close()

    for sample_size, results_by_year in results_by_sample_size_dict.items():
        output_list = []
        for year, (oii_values, _, _) in results_by_year.items():
            if len(oii_values) >= 1 and oii_values[0] is not None:
                mean = np.mean(oii_values)
                ci_lower = np.percentile(oii_values, 2.5)
                ci_upper = np.percentile(oii_values, 97.5)
                output_list.append({
                    'Year': year,
                    'Mean_OII': mean,
                    'CI_Lower': ci_lower,
                    'CI_Upper': ci_upper
                })

        result_df = pd.DataFrame(output_list)
        result_df.to_csv(f'{filename_prefix}_{sample_size}_results.csv', index=False)


# ================== 5. ==================
if __name__ == "__main__":

    data_file = "filtered_living_data.csv"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    years = list(range(350, 2021))
    sample_sizes = [100, 1000]
    num_bootstrap_samples = 500

    df = load_data(data_file)

    relations = {
        'child': df[df['Relation'] == 'child'],
        'student': df[df['Relation'] == 'student']
    }
    for rel_name, df_rel in relations.items():
        print(f"\nProcessing relation: {rel_name}")

        results_by_sample_size = {
            100: {},
            1000: {},
            'full': {}
        }

        print("Processing sample size 100")
        for year in years:
            oii_values, total_people, child_people = process_year(df_rel, year, sample_size=100,
                                                                  num_bootstrap_samples=num_bootstrap_samples)
            results_by_sample_size[100][year] = (oii_values, total_people, child_people)

        print("Processing sample size 1000")
        for year in years:
            oii_values, total_people, child_people = process_year(df_rel, year, sample_size=1000,
                                                                  num_bootstrap_samples=num_bootstrap_samples)
            results_by_sample_size[1000][year] = (oii_values, total_people, child_people)

        print("Processing full data")
        for year in years:
            oii_values, total_people, child_people = process_year(df_rel, year, sample_size=None)
            results_by_sample_size['full'][year] = (oii_values, total_people, child_people)

        prefix = f"{output_dir}/results_{rel_name}"
        save_results_and_plot(results_by_sample_size, prefix)

    print("\nAll processing completed.")
