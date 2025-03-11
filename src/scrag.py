#%%
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from joblib import Parallel, delayed
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain_milvus import Milvus
from pymilvus import DataType
from tqdm import tqdm

from common import (
    Nv2Embedding,
    SpladeEmbedding,
    distribute_budget,
    gen_user_prompt_few,
    gen_user_prompt_full,
    is_doc_relevant,
    run_prompt,
    tqdm_joblib,
)

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

#%%
INPUT_POST = "..."
DATE_CAP = "2025-02-27T00:00:00.000Z"
LLM_MODEL = "llama3.3-70b"
# %%
G = pd.read_pickle('../data/graph/G.pkl')
embs = pd.read_pickle('../data/intermediates/all_doc_nv2_inst_umap128.pkl')
emb_umapper = pd.read_pickle('../data/intermediates/all_doc_nv2_inst.umapper')
beliefs = pd.read_pickle('../data/intermediates/all_data_belief_emb.pkl')
#%%
sparse_embedder = SpladeEmbedding()
dense_embedder = Nv2Embedding(instruction="Represent the query for retrieving supporting documents.\n", device_map='cpu')
# dense_embedder = VoyageAIEmbeddings(
#     model='voyage-3-large',
#     api_key=os.getenv('VOYAGE_API_KEY'),
#     batch_size=128,
# )
#%%
sparse_article_vs = Milvus(
    embedding_function=sparse_embedder,
    collection_name="news_splade2",
    connection_args={"uri": "http://localhost:19530"},
    auto_id=True,
    primary_field='id',
    text_field='doc',
    vector_field='embedding',
    index_params={
        "field_name": "embedding",
        "index_type": "SPARSE_INVERTED_INDEX",
        'metric_type': "IP",
        "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
    },
    search_params={
        'metric_type': "IP",
        'params': {},
    },
    vector_schema={
        "dtype": DataType.SPARSE_FLOAT_VECTOR
    },
    metadata_schema={
        "data_source": {
            'dtype': DataType.ARRAY, 
            'element_type': DataType.VARCHAR, 
            'max_capacity': 8, 
            'max_length': 32,
        }
    }
)
similar_post_vs = Milvus(
    embedding_function=dense_embedder,
    collection_name='docs_nv2_inst2',
    connection_args={"uri": "http://localhost:19530"},
    primary_field='id',
    text_field='doc',
    vector_field='embedding'
)
# %%
query_sparse_vec = sparse_embedder.embed_query(INPUT_POST)
query_dense_vec = dense_embedder.embed_query(INPUT_POST)
sparse_articles = sparse_article_vs.similarity_search_by_vector(
    query_sparse_vec,
    k=5,
    expr=f'published <= "{DATE_CAP}"',
)
similar_posts = similar_post_vs.similarity_search_by_vector(
    query_dense_vec,
    k=50,
    expr='created_at <= "{DATE_CAP}" AND engagement_in_degree > 0',
)
#%%
with tqdm_joblib(tqdm(desc="Is doc relevant", total=len(similar_posts))) as progress_bar:
    rets = Parallel(n_jobs=32)(delayed(is_doc_relevant)(LLM_MODEL, INPUT_POST, p.page_content) for p in similar_posts)
# %%
relevant_similar_posts = [p.metadata['id'] for p, r in zip(similar_posts, rets) if r]
print('Number of relevant similar posts:', len(relevant_similar_posts))
#%%
emb_list = []
responses = []
for i, similar_post in enumerate(relevant_similar_posts): # Iterate over similar posts:
    for engagement in G.predecessors(similar_post):       # Get the actual responses
        if G.nodes[engagement]['clean_word_count'] < 6:
            continue
        responses.append(engagement)
        emb_list.append(embs[G.nodes[engagement]['doc_enc']])
    if i >= 2 and len(responses) >= 60:
        break
order = maximal_marginal_relevance(emb_umapper.transform([query_dense_vec]), emb_list, k=50)
relevant_responses = [responses[ii] for ii in order]
print('Number of traced engagements:', len(relevant_responses))
# %%
res_embs = np.array([embs[G.nodes[vv]['doc_enc']] for vv in relevant_responses])
belief_embs = np.array([beliefs[vv] for vv in relevant_responses])  #BELIEF PART
res_embs = np.hstack([res_embs, belief_embs])
hdb1 = HDBSCAN(min_cluster_size=2)
hdb1.fit(res_embs)
hdb2 = HDBSCAN(min_cluster_size=5)
hdb2.fit(res_embs)
hdb3 = HDBSCAN(min_cluster_size=6)
hdb3.fit(res_embs)
if len(set(hdb3.labels_)) >= 4:
    hdb = hdb3
elif len(set(hdb2.labels_)) >= 4:
    hdb = hdb2
else:
    hdb = hdb1

clusters = defaultdict(list)
for res, lab in zip(relevant_responses, hdb.labels_):
    clusters[int(lab)].append(res)
for cid, cl in clusters.items():
    if cid >= 0:
        medoid = hdb.weighted_cluster_medoid(cid)
        clusters[cid] = sorted(cl, key=lambda x: np.linalg.norm(
            np.hstack([embs[G.nodes[x]['doc_enc']], beliefs[x]]) - medoid
        ))
if -1 in clusters:
    if len(clusters[-1]) < 2:
        clusters.pop(-1)
    elif len(clusters[-1]) >= 6:
        univ = []
        pro = []
        anti = []
        for res in clusters[-1]:
            if beliefs[res].sum() < 0.01:
                univ.append(res)
            elif beliefs[res].argmax() % 2 == 0:
                pro.append(res)
            else:
                anti.append(res)
        orig_pro_len = len(pro)
        pro.extend(univ[0::2])
        anti.extend(univ[1::2])
        
        if len(pro) >= 3 and len(anti) >= 3:
            clusters[-1] = []
            clusters[-2] = []
            cnt = 0
            for di in range(max(len(pro), len(anti))):
                if di < len(pro):
                    clusters[-1].append(pro[di])
                    cnt += 1
                    if cnt >= 12:
                        break
                if di < len(anti):
                    clusters[-2].append(anti[di])
                    cnt += 1
                    if cnt >= 12:
                        break
    # clusters[-1] = clusters[-1][:12]
csizes = []
csamples = []
for cid in range(min(clusters), max(clusters) + 1):
    csizes.append(len(clusters[cid]))
    csamples.append(clusters[cid])
dist = distribute_budget(30, csizes)
assert len(dist) == len(csamples)
#%%
SYSTEM_PROMPT = """You are an X (Twitter) user who browses specific content and likes to interact. You know world news and geopolitical matters and like to engage with online communications with people from your community/group. You can read multiple languages, but you ALWAYS write posts in English.

You will be given news article snippets and several knowledge graph relations for background knowledge, and you will write a reply/response tweet to a new user-provided post in your response. You will also be provided examples of how other users in your community respond in a context similar to the new post.

Your writing style SHOULD BE similar to an X (Twitter) user.
"""
the_kg = ""
prompts = []
for cidx, (ncnt, samples) in enumerate(zip(dist, csamples)):
    other_cand = np.argsort(-np.array(dist)).tolist()
    other_cand.remove(cidx)
    if len(other_cand) < 2:
        USER_PROMPT = gen_user_prompt_few(sparse_articles, INPUT_POST, [G.nodes[n]['doc_enc'] for n in samples], the_kg)
    else:
        other_samples = []
        otherd = 0
        while len(other_samples) < 2:
            for oidx in other_cand:
                other_samples.append(csamples[oidx][otherd])
                if len(other_samples) >= 2:
                    break
            otherd += 1
        USER_PROMPT = gen_user_prompt_full(sparse_articles, INPUT_POST, [G.nodes[n]['doc_enc'] for n in samples], [G.nodes[n]['doc_enc'] for n in other_samples], the_kg)
    for reply_no in range(ncnt):
        prompts.append({
            "model": LLM_MODEL,
            "messages": [
                {"role": "developer" if 'gpt-4o' in LLM_MODEL else "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ] if LLM_MODEL != 'gemma' else [
                {"role": "user", "content": SYSTEM_PROMPT + '\n' + USER_PROMPT}
            ],
            "temperature": 1.2,
            "max_completion_tokens": 100,
        })
#%%
with tqdm_joblib(tqdm(desc="Generating", total=len(prompts))) as progress_bar:
    rets = Parallel(n_jobs=32)(delayed(run_prompt)(p) for p in prompts)
# %%
