# SCRAG: Social Computing-Based Retrieval-Augmented Generation for Community Response Forecasting

## Overview

SCRAG is a predictive framework designed to forecast community responses to real or hypothetical social media posts. It integrates large language models (LLMs) with Social Computing-Based Retrieval-Augmented Generation (RAG) techniques rooted in social computing. By retrieving historical responses and external knowledge, SCRAG generates diverse, realistic, and contextually grounded predictions of community reactions.

## Features

- Social Computing-Inspired Retrieval-Augmented Generation (RAG):
  - Historical Response Retrieval Module: Captures ideological, semantic, and emotional aspects of past responses to inform predictions.
  - External Knowledge Retrieval Module: Integrates external knowledge from news sources and knowledge graphs to provide up-to-date context.
- Modular and Adaptable: Compatible with various embedding models and LLMs for flexibility.
- Comprehensive Evaluation: Tested across six real-world scenarios on the X platform (formerly Twitter), demonstrating over 10% improvements in key evaluation metrics.

## Running the framework

### Preprocessing

Create a Python 3.10 environment and install the required packages:
```bash
pip install -r requirements.txt
```

Initialize the Milvus DB by running:
```bash
cd milvus
sudo sh standalone_embed.sh start
```

In `src/preprocess`:
```bash
cd src/preprocess
python preprocess_docs.py
python preprocess_belief_emb.py
# <Run commands printed by the previous command>
...
python extract_kg.py \
  --input "../../data/news/news.jsonl" \ # Or custom name for JSONL
  [--joblib 32]
```

In `src/emb_scripts`:
```bash
cd src/emb_scripts
# Run selected embedding (VoyageAI for example) for each file in data/social_media_docs
python text_emb/emb_voyage_int.py --input <Doc path>
# Gather all results
python gather_emb_voyage.py
python umap_emb.py \
  --input "../../data/intermediates/all_doc_voyage_inst.pkl"
```
```bash
# Push to Milvus
python push_docs.py \
  --docs "../../data/graph/G.pkl" \
  --embs "../../data/intermediates/all_doc_voyage_inst.pkl" \
  --colname docs_voyage_inst # Or custom collection name for Milvus
python emb_and_push_splade_news.py \
  --input "../../data/news/news.jsonl" \ # Or custom name for JSONL
  [--colname news_splade] [--device cpu] # Or custom collection name for Milvus and device
python emb_and_push_splade_kg.py \
  --input "../../data/kg/kg.pkl" \ # Or custom name for KG
  [--colname news_kg] [--device cpu] # Or custom collection name for Milvus and device
```

Set the necessary environment variables including `OPENAI_API_KEY` and `VOYAGE_API_KEY`.

### Forecasting

Run the framework by modifying `src/scrag.py`:
```bash
cd src
python scrag.py
```
