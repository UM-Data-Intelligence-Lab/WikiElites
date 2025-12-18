import pandas as pd
import math
from collections import Counter

def load_data(file_path):
    df = pd.read_csv(file_path)

    df = df.dropna(subset=[
        'Node1_Birth', 'Node2_Birth',
        'Node1_Death', 'Node2_Death',
        'Node1_occ_level3', 'Node2_occ_level3',
        'Node1_Country', 'Node2_Country'
    ])

    df['Node1_Birth'] = df['Node1_Birth'].astype(int)
    df['Node2_Birth'] = df['Node2_Birth'].astype(int)
    df['Node1_Death'] = df['Node1_Death'].fillna(2025).astype(int)
    df['Node2_Death'] = df['Node2_Death'].fillna(2025).astype(int)

    df['Country'] = df['Country'].str.replace(r'^Old_regimes_in_/_of_', '', regex=True)

    return df

# ================== 2.  OII ==================
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

# ================== 3. Year ==================
def process_year(df_relation, year):
    df_year = df_relation[
        ((df_relation['Node1_Birth'] <= year) & (df_relation['Node1_Death'] > year)) &
        ((df_relation['Node2_Birth'] <= year) & (df_relation['Node2_Death'] > year))
    ]

    countries = set(df_year['Node1_Country']).union(set(df_year['Node2_Country']))
    results = []

    for country in countries:
        df_country = df_year[
            (df_year['Node1_Country'] == country) | (df_year['Node2_Country'] == country)
        ]

        oii = calculate_oii(df_country)
        occupation_count = len(set(df_country['Node1_occ_level3']).union(set(df_country['Node2_occ_level3'])))
        relation_count = len(df_country)

        if oii is not None:
            results.append({
                'Year': year,
                'Country': country,
                'OII': oii,
                'Occupation_Count': occupation_count,
                'Relation_Count': relation_count
            })

    return results

# ================== 4.main ==================
if __name__ == "__main__":
    data_file = "filtered_living_data.csv" 
    output_file = "result_OII.csv"
    years = list(range(1800, 2021)) 

    df = load_data(data_file)
    df_child = df[df['Relation'] == 'child']

    all_results = []
    for year in years:
        yearly_results = process_year(df_child, year)
        all_results.extend(yearly_results)

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_file, index=False)
    print(f"✅ Processing completed. Results saved to {output_file}")
