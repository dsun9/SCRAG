#%%
import argparse
import hashlib
import json
import os
import time
from math import ceil
from pathlib import Path

import pandas as pd
from natsort import natsorted
from openai import OpenAI


#%%
def generate_custom_id(text):
    encoded_text = text.encode('utf-8')
    hash_object = hashlib.sha256(encoded_text)
    custom_id = hash_object.hexdigest()
    return custom_id

#%%
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=45000, help='specify batch size')
    parser.add_argument('--inst', type=bool, default=True, action=argparse.BooleanOptionalAction, help='use instruction')
    parser.add_argument('--input_dir', type=str, default='../../docs', help='specify input folder')
    args = parser.parse_args()
    #%%
    doc_db = []
    for fn in Path(args.input_dir).glob('*.pkl'):
        docs = pd.read_pickle(fn)
        doc_db.extend(docs)
    #%%
    preferred_batch_size = ceil(len(doc_db) / ceil(len(doc_db) / args.batch_size))
    instruction = "Given the social media message history and the user's response to that, represent and cluster the document based on the response under the situation.\n" if args.inst else ""
    for i in range(0, len(doc_db), preferred_batch_size):
        docs = [instruction + doc for doc in doc_db[i:i+preferred_batch_size]]
        reqs = [
            {
                "custom_id": generate_custom_id(docs[j]),
                "method": "POST", 
                "url": "/v1/embeddings", 
                "body": {
                    "input": docs[j:j+8], 
                    "model": "text-embedding-3-large", 
                    "dimensions": 1024,
                }
            } for j in range(0, len(docs), 8)
        ]
        with open(f'{args.input_dir}/openai_{"inst" if args.inst else "noinst"}_batch_{i // preferred_batch_size}.jsonl', 'w', encoding='utf-8') as f:
            for r in reqs:
                f.write(json.dumps(r, ensure_ascii=False) + '\n')
    #%%
    for fn in natsorted(Path(args.input_dir).glob('openai_*.jsonl')):
        fnn = fn.with_suffix('.fileid.txt')
        print(fnn)
        if fnn.exists() and fnn.with_suffix('').with_suffix('.jsonl').name == client.files.retrieve(fnn.read_text()).filename:
            print('existing')
        else:
            batch_input_file = client.files.create(
                file=open(fn, "rb"),
                purpose="batch"
            )
            with open(fn.with_suffix('.fileid.txt'), 'w') as f2:
                f2.write(batch_input_file.id)
    # %%
    for fn in natsorted(Path(args.input_dir).glob('openai_*.jsonl')):
        if fn.name.endswith('.embjsonl'):
            continue
        print(fn)
        if fn.with_suffix('.batchid.txt').exists():
            batch_id = fn.with_suffix('.batchid.txt').read_text()
            monitor_batch = client.batches.retrieve(batch_id)
            if monitor_batch.status == 'failed':
                print('previous run found, batch failed, deleting')
                fn.with_suffix('.batchid.txt').unlink()
        if not fn.with_suffix('.batchid.txt').exists():
            batch_input_file_id = fn.with_suffix('.fileid.txt').read_text()
            batch = client.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/embeddings",
                completion_window="24h",
                metadata={
                    "description": fn.name
                }
            )
            with open(fn.with_suffix('.batchid.txt'), 'w') as f2:
                f2.write(batch.id)
            batch_id = batch.id
        else:
            print('existing')
            batch_id = fn.with_suffix('.batchid.txt').read_text()
        monitor_batch = client.batches.retrieve(batch_id)
        while True:
            if monitor_batch.status == 'completed':
                print('batch completed')
                break
            elif monitor_batch.status == 'failed' or monitor_batch.status == 'expired' or monitor_batch.status == 'cancelled':
                print('batch failed/cancelled')
                break
            time.sleep(5)
            monitor_batch = client.batches.retrieve(batch_id)
        if monitor_batch.output_file_id:
            with open(fn.with_suffix('.resultid.txt'), 'w') as f3:
                f3.write(monitor_batch.output_file_id)
    # %%
    for fn in natsorted(Path(args.input_dir).glob('openai_*.jsonl')):
        print(fn)
        if fn.with_suffix('.batchid.txt').exists():
            batch_id = fn.with_suffix('.batchid.txt').read_text()
            monitor_batch = client.batches.retrieve(batch_id)
            if monitor_batch.status == 'completed':
                print('batch completed')
                if monitor_batch.output_file_id:
                    with open(fn.with_suffix('.resultid.txt'), 'w') as f3:
                        f3.write(monitor_batch.output_file_id)
    # %%
    for fn in natsorted(Path(args.input_dir).glob('openai_*.jsonl')):
        if fn.with_suffix('.resultid.txt').exists():
            print(fn)
            print('done')
            rfn = fn.with_suffix('.embjsonl')
            if rfn.exists():
                print('existing')
                continue
            result_id = fn.with_suffix('.resultid.txt').read_text()
            result = client.files.content(result_id)
            with open(rfn, 'wb') as f:
                f.write(result.content)
    # %%
