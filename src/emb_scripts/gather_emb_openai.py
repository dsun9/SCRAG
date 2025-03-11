#%%
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from natsort import natsorted
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inst', type=bool, default=True, action=argparse.BooleanOptionalAction, help='use instruction')
    args = parser.parse_args()
    args.input_dir = Path('../../data/social_media_docs')

    doc_db = []
    for fn in args.input_dir.glob('*.pkl'):
        docs = pd.read_pickle(fn)
        doc_db.extend(docs)

    emb_results = {}
    instruction = "Given the social media message history and the user's response to that, represent and cluster the document based on the response under the situation.\n" if args.inst else ""
    req_list = {}
    res_list = {}
    for fn in tqdm(natsorted(args.input_dir.glob(f'openai_{"inst" if args.inst else "noinst"}*.embjsonl')), ncols=100):
        dfn = fn.with_suffix('.jsonl')
        with open(dfn, 'r', encoding='utf-8') as f:
            for l in f:
                tmp = json.loads(l)
                req_list[tmp['custom_id']] = tmp
        with open(fn, 'r', encoding='utf-8') as f:
            for l in f:
                tmp = json.loads(l)
                res_list[tmp['custom_id']] = tmp
    for cid in req_list:
        r = req_list[cid]
        s = res_list[cid]
        assert r['custom_id'] == s['custom_id']
        for i, v in enumerate(s['response']['body']['data']):
            assert v['index'] == i
        for doc, emb_body in zip(r['body']['input'], s['response']['body']['data']):
            emb_results[doc[len(instruction):]] = np.array(emb_body['embedding'])

    norms = [c@c.T for c in emb_results.values()]
    print(np.min(norms), np.max(norms))
    print(any(np.isnan(norms)))
    assert len(doc_db) == len(set(doc_db) & set(emb_results))

    pd.to_pickle(emb_results, f'../../data/intermediates/all_doc_openai_{"inst" if args.inst else "noinst"}.pkl')