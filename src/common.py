import contextlib
import os

import joblib
import pandas as pd
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from pymilvus import model as milvus_model
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class QueryEmbeddingFromDict(Embeddings):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        if isinstance(dictionary, str):
            print('Loading embeddings')
            self.dictionary = pd.read_pickle(dictionary)
        elif isinstance(dictionary, dict):
            self.dictionary = dictionary
        else:
            raise Exception("Cannot load embedding")
    
    def embed_documents(self, texts):
        res = [self.dictionary[text] for text in texts]
        return res
    
    def embed_query(self, text):
        return self.dictionary[text]

class SpladeEmbedding(Embeddings):
    def __init__(self, device='cpu', **kwargs):
        super().__init__(**kwargs)
        self.splade_ef = milvus_model.sparse.SpladeEmbeddingFunction(
            model_name="naver/splade-cocondenser-ensembledistil", 
            device=device
        )
    def embed_documents(self, texts):
        outputs = self.splade_ef.encode_documents(texts)
        ret = [{int(x): float(y) for x, y in output.todok().items()} for output in outputs]
        return ret
    def embed_query(self, text):
        return self.embed_documents([text])[0]

class Nv2Embedding(Embeddings):
    def __init__(self, instruction="", device_map='cpu', **kwargs):
        super().__init__(**kwargs)
        self.instruction = instruction
        from transformers import AutoModel
        self.model = AutoModel.from_pretrained('nvidia/NV-Embed-v2', trust_remote_code=True, device_map=device_map)
    def embed_documents(self, texts):
        res = []
        for text in texts:
            res.append(self.model.encode(text, instruction=self.instruction, max_length=32768).clone().detach()[0].cpu().tolist())
        return res
    def embed_query(self, text):
        return self.embed_documents([text])[0]

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def distribute_budget(N, weights):
    k = len(weights)

    if N < k:
        N = k

    total_weight = sum(weights)
    weights = [weight / total_weight for weight in weights]
    total_weight = sum(weights)
    if not abs(total_weight - 1.0) < 1e-6:
        raise ValueError("Weights must sum up to 1.")

    allocations = [1] * k
    remaining_units = N - k  # Units left after initial allocation
    preliminary_allocations = [weight * remaining_units for weight in weights]
    integer_parts = [int(allocation) for allocation in preliminary_allocations]
    fractional_parts = [allocation - int_part for allocation, int_part in zip(preliminary_allocations, integer_parts)]
    allocated_units = sum(integer_parts)
    units_to_allocate = remaining_units - allocated_units
    sorted_indices = sorted(range(k), key=lambda i: -fractional_parts[i])

    for i in sorted_indices:
        if units_to_allocate <= 0:
            break
        integer_parts[i] += 1
        units_to_allocate -= 1
    allocations = [alloc + int_part for alloc, int_part in zip(allocations, integer_parts)]

    return allocations

def run_prompt(prompt):
    @retry(wait=wait_random_exponential(multiplier=1, max=8), stop=stop_after_attempt(160))  
    def inner():
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url='http://128.174.212.200:11435/v1')
        return client.chat.completions.create(**prompt)
    res = inner()
    return res.model_dump()

def is_doc_relevant(LLM_MODEL, post_a, post_b):
    SYSTEM_PROMPT = """You are an X (Twitter) user who browses diverse content and likes to interact. You know world news and geopolitical matters.

When given two posts that are either original posts or replies/responses to another post (context attached), you will answer the question of whether the two posts are related, thinking similarly, or expressing similar beliefs under potentially different scenarios. You will only answer "yes" or "no". When ambiguous, you will answer "no".
"""
    USER_PROMPT = f"""Post A:
\"\"\"
{post_a}
\"\"\"

Post B:
\"\"\"
{post_b}
\"\"\"

Answer:
"""
    @retry(wait=wait_random_exponential(multiplier=1, max=8), stop=stop_after_attempt(8))  
    def inner():
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), base_url='http://localhost:11435/v1')
        return client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            temperature=0.0,
            max_completion_tokens=16,
            extra_body={"guided_choice": ["Yes.", "No."]},
        )
    doc_relevant = inner()
    return 'yes' in doc_relevant.choices[0].message.content.lower()

def gen_user_prompt_full(articles, input_post, samples, other_samples, kg):
    return f"""The following news and entity relations are the latest information for you to know.
<NEWS_ARTICLE_SNIPPETS>
{'\n\n'.join([f"\"\"\"\n{article.page_content}\n\"\"\"" for article in articles[:5]])}
</NEWS_ARTICLE_SNIPPETS>

<KNOWLEDGE_GRAPH>
{kg}
</KNOWLEDGE_GRAPH>


The following are examples of how users in *OTHER* communities respond to posts in a context similar to the new post:
<RESPONSE_EXAMPLES>
{'\n\n'.join([f"\"\"\"\n{n}\n\"\"\"" for n in other_samples[:2]])}
</RESPONSE_EXAMPLES>


The following are examples of how other users in *YOUR* community respond to posts in a context similar to the new post.
<RESPONSE_EXAMPLES>
{'\n\n'.join([f"\"\"\"\n{n}\n\"\"\"" for n in samples[:3]])}
</RESPONSE_EXAMPLES>



New post:
<CONTEXT>
{input_post}
</CONTEXT>

As a member of your community, write your reply/response post DIRECTLY:
"""
#%%
def gen_user_prompt_few(articles, input_post, samples, kg):
    return f"""<NEWS_ARTICLE_SNIPPETS>
{'\n\n'.join([f"\"\"\"\n{article.page_content}\n\"\"\"" for article in articles[:5]])}
</NEWS_ARTICLE_SNIPPETS>

<KNOWLEDGE_GRAPH>
{kg}
</KNOWLEDGE_GRAPH>


The following are examples of how other users in *YOUR* community respond to posts in a context similar to the new post.
<RESPONSE_EXAMPLES>
{'\n\n'.join([f"\"\"\"\n{n}\n\"\"\"" for n in samples[:5]])}
</RESPONSE_EXAMPLES>



New post:
<CONTEXT>
{input_post}
</CONTEXT>

As a member of your community, write your reply/response post DIRECTLY:
"""