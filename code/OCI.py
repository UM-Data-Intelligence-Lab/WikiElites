import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
from multiprocessing import Pool
import time
import os

data = pd.read_csv('Q_R_Q_extended.txt')
data = data[['Node1', 'Relation', 'Node2', 'Node1_occ_level1', 'Node2_occ_level1']]

#kinship 
kinship_relations = ['child', 'father', 'mother', 'sibling', 'spouse']
data['relation_type'] = data['Relation'].apply(lambda r: 'kinship' if r in kinship_relations else 'non-kinship')


occupations = sorted(set(data['Node1_occ_level1']).union(set(data['Node2_occ_level1'])))
print(f"Total occupations: {len(occupations)}")

#  node_to_occ 
unique_nodes = pd.concat([data['Node1'], data['Node2']]).drop_duplicates().reset_index(drop=True)
node_to_occ = pd.DataFrame(unique_nodes, columns=['Node'])
node_to_occ['occ'] = node_to_occ['Node'].map(
    dict(zip(data['Node1'], data['Node1_occ_level1']))
).fillna(
    node_to_occ['Node'].map(dict(zip(data['Node2'], data['Node2_occ_level1'])))
).fillna('unknown')

valid_node_list = node_to_occ['Node'].tolist()
valid_node_to_occ = dict(zip(node_to_occ['Node'], node_to_occ['occ']))

 
# -------------------------------

def compute_total_counts(df, occupations):
    count_matrix = pd.DataFrame(0, index=occupations, columns=occupations)
    for _, row in df.iterrows():
        i = row['Node1_occ_level1']
        j = row['Node2_occ_level1']
        if i in occupations and j in occupations:
            count_matrix.loc[i, j] += 1
    n_i_total = count_matrix.sum(axis=1)
    m_j_total = count_matrix.sum(axis=0)
    return n_i_total, m_j_total


def compute_oci_matrix(df, occupations, n_i_total, m_j_total):
    count_matrix = pd.DataFrame(0, index=occupations, columns=occupations)
    for _, row in df.iterrows():
        i = row['Node1_occ_level1']
        j = row['Node2_occ_level1']
        if i in occupations and j in occupations:
            count_matrix.loc[i, j] += 1

    oci_matrix = pd.DataFrame(0.0, index=occupations, columns=occupations)
    for i in occupations:
        for j in occupations:
            a_ij = count_matrix.loc[i, j]
            ni_total = n_i_total[i]
            mj_total = m_j_total[j]
            if ni_total + mj_total > 0:
                oci_matrix.loc[i, j] = (2 * a_ij) / (ni_total + mj_total)
    return oci_matrix


# -------------------------------
# 3.OCI
# -------------------------------

n_i_total_real, m_j_total_real = compute_total_counts(data, occupations)

real_oci_matrices = {}
for rel_type in ['kinship', 'non-kinship']:
    sub_df = data[data['relation_type'] == rel_type].copy()
    real_oci_matrices[rel_type] = compute_oci_matrix(sub_df, occupations, n_i_total_real, m_j_total_real)

# -------------------------------
# 4. Replacement simulation
# -------------------------------

B = 10 
def run_simulation(b):

    shuffled_occs = np.random.permutation(list(valid_node_to_occ.values()))
    shuffled_map = dict(zip(valid_node_list, shuffled_occs))


    data_shuffled = data.copy()
    data_shuffled['Node1_occ_level1'] = data_shuffled['Node1'].map(shuffled_map)
    data_shuffled['Node2_occ_level1'] = data_shuffled['Node2'].map(shuffled_map)

    data_shuffled.dropna(subset=['Node1_occ_level1', 'Node2_occ_level1'], inplace=True)


    n_i_total_new, m_j_total_new = compute_total_counts(data_shuffled, occupations)

    results = {}
    for rel_type in ['kinship', 'non-kinship']:
        sub_df = data_shuffled[data_shuffled['relation_type'] == rel_type].copy()
        oci_mat = compute_oci_matrix(sub_df, occupations, n_i_total_new, m_j_total_new)
        results[rel_type] = oci_mat

    return results

# -------------------------------
# 5. Multi-process execution simulation
# -------------------------------

start_time = time.time()
print("Start the null-model simulation

...")

with Pool(os.cpu_count()) as pool:
    sim_results = list(tqdm(pool.imap(run_simulation, range(B)), total=B, desc="模拟进度"))

print(f"零模型模拟完成，耗时: {time.time() - start_time:.2f}秒")

# -------------------------------
# 6. all the simulation results
# -------------------------------

null_oci_distributions = {'kinship': {}, 'non-kinship': {}}

for result in tqdm(sim_results, total=B, desc="收集模拟结果"):
    for rel_type in ['kinship', 'non-kinship']:
        oci_mat = result[rel_type]
        for i, j in product(occupations, repeat=2):
            val = oci_mat.loc[i, j]
            if (i, j) not in null_oci_distributions[rel_type]:
                null_oci_distributions[rel_type][(i, j)] = []
            null_oci_distributions[rel_type][(i, j)].append(val)

# -------------------------------
# 7. Calculate the p-value
# -------------------------------

output = []

for rel_type in ['kinship', 'non-kinship']:
    real_matrix = real_oci_matrices[rel_type]
    null_dict = null_oci_distributions[rel_type]

    for i, j in product(occupations, repeat=2):
        real_val = real_matrix.loc[i, j]
        null_vals = null_dict.get((i, j), [np.nan] * B)  # 如果缺失则补 NaN

        while len(null_vals) < B:
            null_vals.append(np.nan)


        count_ge = sum(1 for val in null_vals if not np.isnan(val) and val >= real_val)
        p = (count_ge + 1) / (B + 1)  
        row = {
            's': i,
            'd': j,
            'r': rel_type,
            'OCI': real_val,
            'p': p
        }

        for b_idx in range(B):
            row[f'OCIB{b_idx + 1}'] = null_vals[b_idx]

        output.append(row)

# -------------------------------

result_df = pd.DataFrame(output)
cols = ['s', 'd', 'r', 'OCI'] + [f'OCIB{i + 1}' for i in range(B)] + ['p']
result_df = result_df[cols]

result_df.to_csv('s_d_r_OCI_OCIB1_p.csv', index=False)

