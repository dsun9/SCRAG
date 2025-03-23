import argparse
import pickle
from pathlib import Path

from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8, help='specify batch size')
    parser.add_argument('--inst', type=bool, default=True, action=argparse.BooleanOptionalAction, help='use instruction')
    parser.add_argument('--input', type=str, required=True, help='specify input df')
    args = parser.parse_args()

    args.input = Path(args.input).resolve()
    out_file = args.input.with_suffix('.nv_inst.emb' if args.inst else '.nv_noinst.emb')
    if out_file.exists():
        print(f'Output file {out_file} already exists. Skipping...')
        return
    print(out_file)

    with open(args.input, 'rb') as f:
        doc_db = pickle.load(f)
    # Create an LLM.
    from transformers import AutoModel
    model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True, device_map='auto')

    output_map = {}

    print(f'Processing batch {args.input.stem}...')
    print(f'Number of documents in this batch: {len(doc_db)}')
    print(f'Number of batches: {len(doc_db) / args.batch_size}')
    
    instruction = "Given the social media message history and the user's response to that, represent and cluster the document based on the response under the situation.\n" if args.inst else ""

    for i in tqdm(range(0, len(doc_db), args.batch_size)):
        docs = doc_db[i:i+args.batch_size]
        outputs = model.encode(docs, instruction=instruction, max_length=32768)
        outputs = outputs.detach().cpu().numpy()

        for i in range(len(outputs)):
            output = outputs[i]
            output_map[docs[i]] = output

    with open(out_file, "wb") as f:
        pickle.dump(output_map, f)

if __name__ == '__main__':
    main()
