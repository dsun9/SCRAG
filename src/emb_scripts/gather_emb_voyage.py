#%%
import argparse
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
    for fn in tqdm(natsorted(args.input_dir.glob(f'doc_set_*.voyage_{"inst" if args.inst else "noinst"}.emb')), ncols=100):
        vals = pd.read_pickle(fn)
        emb_results.update({k[len(instruction):]: v for k, v in vals.items()})

    norms = [c@c.T for c in emb_results.values()]
    print(np.min(norms), np.max(norms))
    print(any(np.isnan(norms)))
    assert len(doc_db) == len(set(doc_db) & set(emb_results))

    pd.to_pickle(emb_results, f'../../data/intermediates/all_doc_voyage_{"inst" if args.inst else "noinst"}.pkl')