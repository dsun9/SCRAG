#%%
import argparse
from pathlib import Path

import pandas as pd
import umap

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=False, default=None)
    args = parser.parse_args()
    args.input = Path(args.input)
    if args.output_dir is None:
        args.output_dir = args.input.parent
    args.output_dir = Path(args.output_dir)
    
    mapper_fn = args.output_dir / args.input.with_suffix('.umapper').name
    reduced_fn = args.output_dir / args.input.with_name(args.input.stem + '_umap128.pkl').name

    embs = pd.read_pickle(args.input)
    ks = []
    vs = []
    for k, v in embs.items():
        ks.append(k)
        vs.append(v)
    mapper = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=128, n_jobs=20).fit(vs)
    pd.to_pickle(mapper, mapper_fn)
    
    new_embs = {}
    for ks, newv in zip(ks, mapper.embedding_):
        new_embs[ks] = newv
    pd.to_pickle(new_embs, reduced_fn)