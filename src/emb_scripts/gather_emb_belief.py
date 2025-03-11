#%%
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm


#%%
def normalize_points(points, quantile=0.95, scale=8):
    dists = np.linalg.norm(points, axis=1)
    if np.all(dists == 0):
        return points.copy()
    nonzero_dists = dists[dists > 0]
    R = np.percentile(nonzero_dists, quantile * 100)
    new_dists = (2 / np.pi) * np.arctan((np.pi / 2) * (dists / R))
    new_points = np.zeros_like(points)
    nonzero = dists > 0
    new_points[nonzero] = (points[nonzero].T / dists[nonzero]).T * new_dists[nonzero, np.newaxis]
    x = new_points[:, 0]
    y = new_points[:, 1]
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x) - np.pi / 4
    new_theta = np.tanh(theta * scale) * np.pi / 4 + np.pi / 4
    new_x = r * np.cos(new_theta)
    new_y = r * np.sin(new_theta)
    new_points = np.vstack([new_x, new_y]).T
    return new_points
#%%
def pad_np_array(arr, n, dataset_index):
    padded = np.zeros(n * 2, dtype=arr.dtype)
    start_idx = dataset_index * 2
    padded[start_idx:start_idx + 2] = arr
    return padded
#%%
def propagate(G):
    new_cnt = 1
    while new_cnt > 0:
        for n, d in G.nodes(data=True):
            if 'emb' in d:
                for nei in G.neighbors(n):
                    if 'emb' not in G.nodes[nei]:
                        # G.nodes[nei]['col'].append(d['emb'])
                        if d['emb'][0] < d['emb'][1]:
                            G.nodes[nei]['A'].append(d['emb'])
                        else:
                            G.nodes[nei]['B'].append(d['emb'])
        new_cnt = 0
        for n, d in G.nodes(data=True):
            if 'emb' not in d and (len(d['A']) > 0 or len(d['B']) > 0):
                if len(d['A']) > 0 and len(d['B']) > 0:
                    if np.linalg.norm(d['A']).sum() > np.linalg.norm(d['B']).sum():
                        d['emb'] = np.mean(d['A'], axis=0)
                    else:
                        d['emb'] = np.mean(d['B'], axis=0)
                    new_cnt += 1
                elif len(d['A']) > 0:
                    d['emb'] = np.mean(d['A'], axis=0)
                    new_cnt += 1
                elif len(d['B']) > 0:
                    d['emb'] = np.mean(d['B'], axis=0)
                    new_cnt += 1
        print(new_cnt)
# %%
all_belief_emb = defaultdict(list)
datasets = sorted(map(lambda x: x.stem, Path('../../data/separated_for_belief').glob('*.pkl')))
for i, d_name in enumerate(datasets):
    # d_name = 'covid'
    data = pd.read_pickle(f'../../data/separated_for_belief/{d_name}.pkl')
    df, df_filter = data['df'], data['df_filter']
    G, G_filter = data['G'], data['filter_G']
    index2id = data['index2id']
    y2d = normalize_points(np.load(f'../../data/separated_for_belief/{d_name}_2.npy'))
    for n, e in zip(G_filter, y2d):
        G.nodes[n]['emb'] = e
    for n, d in G.nodes(data=True):
        d['A'] = []
        d['B'] = []
    propagate(G)
    for idx, expanded in tqdm(index2id.items(), ncols=100):
        for e in expanded:
            all_belief_emb[e].append(pad_np_array(G.nodes[idx].get('emb', np.zeros(2, dtype=np.float32)), len(datasets), i))
# %%
for k, v in tqdm(all_belief_emb.items(), ncols=100):
    all_belief_emb[k] = np.max(v, axis=0)
# %%
pd.to_pickle(all_belief_emb, '../../data/intermediates/all_data_belief_emb.pkl')
# %%
