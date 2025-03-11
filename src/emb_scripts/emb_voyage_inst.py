import argparse
import os
import pickle
import time
from pathlib import Path

import numpy as np
import voyageai
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=128, help='specify batch size')
    parser.add_argument('--inst', type=bool, default=True, action=argparse.BooleanOptionalAction, help='use instruction')
    parser.add_argument('--input', type=str, required=True, help='specify input df')
    args = parser.parse_args()
    
    vo = voyageai.Client(api_key=os.getenv('VOYAGE_API_KEY'))
    @retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(16))  
    def embed_with_backoff(**kwargs):
        return vo.embed(**kwargs)

    args.input = Path(args.input).resolve()
    out_file = args.input.with_suffix('.voyage_inst.emb' if args.inst else '.voyage_noinst.emb')
    if out_file.exists():
        print(f'Output file {out_file} already exists. Skipping...')
        return

    with open(args.input, 'rb') as f:
        doc_db = pickle.load(f)

    output_map = {}

    print(f'Processing batch {args.input.stem}...')
    print(f'Number of documents in this batch: {len(doc_db)}')
    print(f'Number of batches: {len(doc_db) / args.batch_size}')
    
    instruction = "Given the social media message history and the user's response to that, represent and cluster the document based on the response under the situation.\n" if args.inst else ""

    for i in tqdm(range(0, len(doc_db), args.batch_size)):
        docs = [instruction + doc for doc in doc_db[i:i+args.batch_size]]
        outputs = np.array(embed_with_backoff(texts=docs, model="voyage-3-large").embeddings)

        for i in range(len(outputs)):
            output = outputs[i]
            output_map[docs[i]] = output
        time.sleep(0.1)

    with open(out_file, "wb") as f:
        pickle.dump(output_map, f)

if __name__ == '__main__':
    main()
