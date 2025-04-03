import contextlib
import json
import os
import re

import joblib
import pandas as pd
from langchain_core.embeddings import Embeddings
from openai import OpenAI
from pydantic import BaseModel
from pymilvus import model as milvus_model
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


class IdeologySummary(BaseModel):
    ideology: str
    description: str
    reasoning: str

class QueryEmbeddingFromDict(Embeddings):
    def __init__(self, dictionary, **kwargs):
        super().__init__(**kwargs)
        if isinstance(dictionary, str):
            print("Loading embeddings")
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
    def __init__(self, device="cpu", **kwargs):
        super().__init__(**kwargs)
        self.splade_ef = milvus_model.sparse.SpladeEmbeddingFunction(
            model_name="naver/splade-cocondenser-ensembledistil", device=device
        )

    def embed_documents(self, texts):
        outputs = self.splade_ef.encode_documents(texts)
        ret = [{int(x): float(y) for x, y in output.todok().items()} for output in outputs]
        return ret

    def embed_query(self, text):
        return self.embed_documents([text])[0]


class Nv2Embedding(Embeddings):
    def __init__(self, instruction="", device_map="cpu", **kwargs):
        super().__init__(**kwargs)
        self.instruction = instruction
        from transformers import AutoModel

        self.model = AutoModel.from_pretrained("nvidia/NV-Embed-v2", trust_remote_code=True, device_map=device_map)

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


def strip_enclosing_quotes(s: str) -> str:
    s = s.strip()
    if len(s) >= 2 and (s[0] == s[-1]) and (s[0] in ("'", '"')):
        return s[1:-1]
    return s.strip()


def process_mentions(tweet, keep_n=0):
    def replace_mention_group(match):
        mentions = re.findall(r"@\w+", match.group())
        num_mentions = len(mentions)

        if keep_n == 0:
            replace_count = num_mentions
        else:
            if num_mentions <= keep_n:
                return match.group()  # No replacement needed
            replace_count = num_mentions - keep_n
        # Determine replacement string

        if replace_count == 1:
            replacement = "@<USER_MENTION>"
        else:
            replacement = "@<USER_MENTION_LIST>"
        if keep_n == 0:
            return replacement
        else:
            # Retain the first 'keep_n' mentions and append replacement

            retained = " ".join(mentions[:keep_n])
            return f"{retained} {replacement}"

    # Regex matches any group of mentions (1 or more)

    processed_tweet = re.sub(r"@\w+(?:\s+@\w+)*", replace_mention_group, tweet)
    return processed_tweet


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


def run_prompt(base_url, prompt):
    @retry(wait=wait_random_exponential(multiplier=1, max=8), stop=stop_after_attempt(160))
    def inner():
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
        return client.chat.completions.create(**prompt)

    res = inner()
    return res.model_dump()


def is_doc_relevant(llm_model, base_url, post_a, post_b):
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
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
        return client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "developer" if "gpt-4o" in llm_model else "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            temperature=0.0,
            max_completion_tokens=16,
            extra_body={"guided_choice": ["Yes.", "No."]},
        )

    doc_relevant = inner()
    return "yes" in doc_relevant.choices[0].message.content.lower()

def gen_community_summary(llm_model, base_url, samples):
    SYSTEM_PROMPT = """You are analyzing a collection of tweets from a group of people who generally share a common ideology or belief system. Each tweet may either be a standalone post or a reply to another tweet. If it is a reply, the original tweet will be provided within <CONTEXT></CONTEXT> tags, followed by the user's response after "RESPONSE:".

Your task is determining the broad ideological category that best represents the group's overall stance. The ideological classification should be general and widely recognized. Do not use niche subcategories or specific affiliations. You should also generate a short description of the ideological category so that people can impersonate the group members later. This description *SHOULD* be general and widely recognized description of the ideological category, *DO NOT* mention specific things related to the sample tweets.

# Guidelines:

- Prioritize how users respond in their RESPONSE: to infer their stance if the context exists.
- Identify recurring themes, attitudes, and perspectives across multiple responses.
- If multiple ideologies are present, choose the most dominant three based on frequency and consistency.
- Your output should be concise and objective.
- The tweets may be multilingual, but the output should summarize in English.

# Output Format:
{
    "ideology": <Your summary here>, 
    "description": <Your concise description of the ideology category/categories>, 
    "reasoning": <reasoning in string> 
}
"""
    USER_PROMPT = f"""Here are the sample tweets:
{'\n\n'.join([f"\"\"\"\n{n}\n\"\"\"" for n in samples])}
"""
    json_schema = IdeologySummary.model_json_schema()
    @retry(wait=wait_random_exponential(multiplier=1, max=8), stop=stop_after_attempt(8))
    def inner():
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), base_url=base_url)
        return json.loads(client.chat.completions.create(
            model=llm_model,
            messages=[
                {"role": "developer" if "gpt-4o" in llm_model else "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": USER_PROMPT}
            ],
            temperature=0.0,
            max_completion_tokens=512,
            extra_body={"guided_json": json_schema},
        ).choices[0].message.content)
    return inner()

def gen_user_prompt(articles, kg, samples, community_summary, input_post):
    return f"""The following news and entity relations are the latest information for you to know.
<NEWS_ARTICLE_SNIPPETS>
{'\n\n'.join([f"\"\"\"\n{article.page_content}\n\"\"\"" for article in articles])}
</NEWS_ARTICLE_SNIPPETS>

<KNOWLEDGE_GRAPH>
{kg}
</KNOWLEDGE_GRAPH>


The following are examples of how other users in *YOUR* community respond to posts in a context similar to the new post.
<RESPONSE_EXAMPLES>
{'\n\n'.join([f"\"\"\"\n{n}\n\"\"\"" for n in samples])}
</RESPONSE_EXAMPLES>

Your community is {community_summary['ideology']}, described as {community_summary['description']}.


New post:
<CONTEXT>
{input_post}
</CONTEXT>

As a member of your community, post your reply/response DIRECTLY:
"""
