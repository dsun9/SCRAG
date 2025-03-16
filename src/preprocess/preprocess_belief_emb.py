#%%
import sys
from pathlib import Path  # if you haven't already done so

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass
from collections import defaultdict

import networkx as nx
import pandas as pd
from tqdm import tqdm
import json

from utils.clean_text import clean_word_bag
#%%
stopwords = list(map(str.strip, Path('../utils/stopwords_en.txt').read_text().strip().split('\n')))
# %%
print('Loading data')
all_data = {}
with open('../../data/social_media/tweets.jsonl', 'r', encoding='utf-8') as f:
    for l in f:
        tmp = json.loads(l)
        all_data[tmp['id']] = tmp
#%%
for k, v in tqdm(all_data.items(), ncols=100):
    bag = clean_word_bag(v['text'])
    v['index_text'] = bag
#%%
print('Generating dataframes')
for k, v in tqdm(all_data.items(), ncols=100):
    v.pop('created_at')
    v.pop('entities')
    v.pop('referenced_tweets')
    v.pop('text')
data_sep = defaultdict(list)
for k, v in tqdm(all_data.items(), ncols=100):
    for d in v['data_source']:
        data_sep[d].append(v)
#%%
for k, v in tqdm(data_sep.items(), ncols=100):
    for vv in v:
        if 'data_source' in vv:
            vv.pop('data_source')
        if 'keyN' not in vv:
            vv['keyN'] = len(vv['index_text'])
        if isinstance(vv['index_text'], list):
            vv['index_text'] = ' '.join(vv['index_text'])
#%%
df_sep = {}
for k, v in tqdm(data_sep.items(), ncols=100):
    df_sep[k] = pd.DataFrame(v)
    tweet_counts_df = df_sep[k].groupby(["index_text"]).size().reset_index(name="tweet_counts")
    user_counts_df = df_sep[k].groupby(["author_id"]).size().reset_index(name="user_counts")
    conv_counts_df = df_sep[k].groupby(["conversation_id"]).size().reset_index(name="conv_counts")
    df_sep[k] = df_sep[k].merge(tweet_counts_df, left_on="index_text", right_on="index_text")
    df_sep[k] = df_sep[k].merge(user_counts_df, left_on="author_id", right_on="author_id")
    df_sep[k] = df_sep[k].merge(conv_counts_df, left_on="conversation_id", right_on="conversation_id")
    df_sep[k]['post_counts'] = df_sep[k].apply(lambda x: max(x.tweet_counts, x.conv_counts), axis=1)
    df_sep[k]['consolid_id'] = df_sep[k].apply(lambda x: x.index_text if x.tweet_counts > x.conv_counts else x.conversation_id, axis=1)
# %%
df_sep_filter = {}
dataset_filter_settings = {
    k: (1, 1, 2)
    for k in df_sep.keys()
}
for k, (kp, ka, kn) in tqdm(dataset_filter_settings.items(), ncols=100):
    df = df_sep[k]
    df_filter = df[(df.post_counts > kp) & (df.user_counts > ka) & (df.keyN > kn)]
    df_sep_filter[k] = df_filter
    print(k, df_filter.author_id.nunique() + df_filter.consolid_id.nunique())
# %%
full_G = {}
for k, v in tqdm(df_sep.items(), ncols=100):
    G = nx.Graph()
    G.add_nodes_from(sorted(v['author_id'].unique()))
    G.add_nodes_from(sorted(v['consolid_id'].unique()))
    e = [(x[0], x[1]) for x in v[['author_id', 'consolid_id']].values]
    e.sort()
    G.add_edges_from(e)
    full_G[k] = G
    print(k, G)
# %%
filter_G = {}
for k, v in tqdm(df_sep_filter.items(), ncols=100):
    G = nx.Graph()
    G.add_nodes_from(sorted(v['author_id'].unique()))
    G.add_nodes_from(sorted(v['consolid_id'].unique()))
    e = [(x[0], x[1]) for x in v[['author_id', 'consolid_id']].values]
    e.sort()
    G.add_edges_from(e)
    filter_G[k] = G
    print(k, G)
# %%
outputs = {}
for k in tqdm(df_sep, ncols=100):
    outputs[k] = {
        'setting': dataset_filter_settings[k],
        'df': df_sep[k],
        'df_filter': df_sep_filter[k],
        'G': full_G[k],
        'filter_G': filter_G[k],
        'index2id': df_sep[k].groupby('consolid_id').id.apply(list).to_dict()
    }
# %%
Path('../../data/separated_for_belief').mkdir(parents=True, exist_ok=True)
for k, v in tqdm(outputs.items(), ncols=100):
    pd.to_pickle(v, '../../data/separated_for_belief/{}.pkl'.format(k))
# %%
for k in outputs:
    print(f'python src/emb_scripts/belief_emb/emb_belief.py --dim 2 --input data/separated_for_belief/{k}.pkl --output data/separated_for_belief/{k}_2.npy')
# %%
