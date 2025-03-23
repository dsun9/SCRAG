import os
from collections import defaultdict

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from joblib import Parallel, delayed
from langchain_core.vectorstores.utils import maximal_marginal_relevance
from langchain_milvus import Milvus
from langchain_voyageai import VoyageAIEmbeddings
from pymilvus import DataType
from tqdm import tqdm

from common import (
    Nv2Embedding,
    SpladeEmbedding,
    strip_enclosing_quotes,
    process_mentions,
    distribute_budget,
    gen_user_prompt,
    is_doc_relevant,
    run_prompt,
    tqdm_joblib,
)

SYSTEM_PROMPT = """You are an X (Twitter) user who browses specific content and likes to interact. You know world news and geopolitical matters and like to engage with online communications with people from your community/group. You can read multiple languages, but you ALWAYS write posts in English.

You will be given news article snippets and several knowledge graph relations for background knowledge, and you will write a reply/response tweet to a new user-provided post in your response. You will also be provided examples of how other users in your community respond in a context similar to the new post.

Your writing style SHOULD BE similar to an X (Twitter) user.
"""


class SCRAG:
    def __init__(self, G, embs, belief_embs, emb_umapper, llm_model, llm_base_url):
        self.G = G
        self.embs = embs
        self.belief_embs = belief_embs
        self.emb_umapper = emb_umapper
        self.llm_model = llm_model
        self.llm_base_url = llm_base_url

        self.sparse_article_vs = None
        self.sparse_relation_vs = None
        self.similar_post_vs = None

    def init_vectorstore(
        self,
        sparse_embedder,
        dense_embedder,
        news_col="news_splade",
        kg_col="news_kg",
        res_col="docs_voyage_inst",
        milvus_uri="tcp://localhost:19530",
    ):
        self.sparse_embedder = sparse_embedder
        self.dense_embedder = dense_embedder
        self.sparse_article_vs = Milvus(
            embedding_function=sparse_embedder,
            collection_name=news_col,
            connection_args={"uri": milvus_uri},
            auto_id=True,
            primary_field="id",
            text_field="doc",
            vector_field="embedding",
            index_params={
                "field_name": "embedding",
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
                "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
            },
            search_params={
                "metric_type": "IP",
                "params": {},
            },
            vector_schema={"dtype": DataType.SPARSE_FLOAT_VECTOR},
            metadata_schema={
                "data_source": {
                    "dtype": DataType.ARRAY,
                    "element_type": DataType.VARCHAR,
                    "max_capacity": 8,
                    "max_length": 32,
                }
            },
        )
        self.sparse_kg_vs = Milvus(
            embedding_function=sparse_embedder,
            collection_name=kg_col,
            connection_args={"uri": milvus_uri},
            auto_id=True,
            primary_field="id",
            text_field="doc",
            vector_field="embedding",
            index_params={
                "field_name": "embedding",
                "index_type": "SPARSE_INVERTED_INDEX",
                "metric_type": "IP",
                "params": {"inverted_index_algo": "DAAT_MAXSCORE"},
            },
            search_params={
                "metric_type": "IP",
                "params": {},
            },
            vector_schema={"dtype": DataType.SPARSE_FLOAT_VECTOR},
        )
        self.similar_post_vs = Milvus(
            embedding_function=dense_embedder,
            collection_name=res_col,
            connection_args={"uri": milvus_uri},
            primary_field="id",
            text_field="doc",
            vector_field="embedding",
        )

    def find_similar_post(self, query, k=40):
        query_dense_vec = self.dense_embedder.embed_query(query)
        similar_posts = self.similar_post_vs.similarity_search_by_vector(
            query_dense_vec,
            k=k,
            expr="engagement_in_degree > 0",
        )
        with tqdm_joblib(tqdm(desc="Is doc relevant", total=len(similar_posts))):
            rets = Parallel(n_jobs=32)(
                delayed(is_doc_relevant)(self.llm_model, self.llm_base_url, query, p.page_content) for p in similar_posts
            )
        relevant_similar_posts = [p.metadata["id"] for p, r in zip(similar_posts, rets) if r]
        if not relevant_similar_posts:
            raise ValueError("No relevant posts found.")
        return query_dense_vec, relevant_similar_posts

    def find_relevant_external_doc(self, query, k=5):
        query_sparse_vec = self.sparse_embedder.embed_query(query)
        sparse_articles = self.sparse_article_vs.similarity_search_by_vector(query_sparse_vec, k=k)
        sparse_relations = self.sparse_kg_vs.similarity_search_by_vector(query_sparse_vec, k=k)
        if not sparse_articles or not sparse_relations:
            raise ValueError("No relevant external knowledge found.")
        return sparse_articles, sparse_relations

    def trace_engagement(self, query_dense_vec, posts, k=30):
        emb_list = []
        responses = []
        for i, similar_post in enumerate(posts):  # Iterate over similar posts:
            for engagement in G.predecessors(similar_post):  # Get the actual responses
                if G.nodes[engagement]["clean_word_count"] < 6:
                    continue
                responses.append(engagement)
                emb_list.append(embs[G.nodes[engagement]["doc_enc"]])
        order = maximal_marginal_relevance(emb_umapper.transform([query_dense_vec]), emb_list, k=k)
        relevant_responses = [responses[ii] for ii in order]
        if not relevant_responses:
            raise ValueError("No relevant responses found.")
        return relevant_responses

    def find_communities(self, relevant_responses):
        res_embs = np.array([self.embs[self.G.nodes[vv]["doc_enc"]] for vv in relevant_responses])
        belief_embs = np.array([self.belief_embs[vv] for vv in relevant_responses])  # BELIEF PART
        res_embs = np.hstack([res_embs, belief_embs])
        hdb = HDBSCAN(min_cluster_size=2)
        hdb.fit(res_embs)

        clusters = defaultdict(list)
        for res, lab in zip(relevant_responses, hdb.labels_):
            clusters[int(lab)].append(res)
        for cid, cl in clusters.items():
            if cid >= 0:
                medoid = hdb.weighted_cluster_medoid(cid)
                clusters[cid] = sorted(
                    cl,
                    key=lambda x: np.linalg.norm(
                        np.hstack([self.embs[self.G.nodes[x]["doc_enc"]], self.belief_embs[x]]) - medoid
                    ),
                )
        if -1 in clusters:
            if len(clusters[-1]) < 2:
                clusters.pop(-1)
            elif len(clusters[-1]) >= 6:
                univ = []
                pro = []
                anti = []
                for res in clusters[-1]:
                    if self.belief_embs[res].sum() < 0.01:
                        univ.append(res)
                    elif self.belief_embs[res].argmax() % 2 == 0:
                        pro.append(res)
                    else:
                        anti.append(res)
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
        csizes = []
        csamples = []
        for cid in range(min(clusters), max(clusters) + 1):
            csizes.append(len(clusters[cid]))
            csamples.append(clusters[cid])
        return csizes, csamples

    def predict_parallel(self, input_post, total=30, n_jobs=32):
        query_dense_vec, relevant_similar_posts = self.find_similar_post(input_post)
        relevant_responses = self.trace_engagement(query_dense_vec, relevant_similar_posts)
        sparse_articles, sparse_relations = self.find_relevant_external_doc(input_post)
        csizes, csamples = self.find_communities(relevant_responses)
        dist = distribute_budget(total, csizes)
        assert len(dist) == len(csamples)

        relations = "\n".join(x.page_content for x in sparse_relations)
        prompts = []
        for cidx, (ncnt, samples) in enumerate(zip(dist, csamples)):
            USER_PROMPT = gen_user_prompt(
                sparse_articles[:5], relations, [G.nodes[n]["doc_enc"] for n in samples[:5]], input_post
            )
            for _ in range(ncnt):
                prompts.append(
                    {
                        "model": self.llm_model,
                        "messages": (
                            [
                                {"role": "developer" if "gpt-4o" in self.llm_model else "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": USER_PROMPT},
                            ]
                            if self.llm_model != "gemma"
                            else [{"role": "user", "content": SYSTEM_PROMPT + "\n" + USER_PROMPT}]
                        ),
                        "seed": np.random.randint(2**32),
                        "temperature": 1.2,
                        "max_completion_tokens": 100,
                        "extra_body": {"seed": np.random.randint(2**32)},
                    }
                )
        with tqdm_joblib(tqdm(desc="Generating", total=len(prompts), ncols=100)):
            rets = Parallel(n_jobs=n_jobs)(delayed(run_prompt)(self.llm_base_url, p) for p in prompts)
        return [process_mentions(strip_enclosing_quotes(r["choices"][0]["message"]["content"])) for r in rets]

    def predict(self, input_post, total=30):
        query_dense_vec, relevant_similar_posts = self.find_similar_post(input_post)
        relevant_responses = self.trace_engagement(query_dense_vec, relevant_similar_posts)
        sparse_articles, sparse_relations = self.find_relevant_external_doc(input_post)
        csizes, csamples = self.find_communities(relevant_responses)
        dist = distribute_budget(total, csizes)
        assert len(dist) == len(csamples)

        relations = "\n".join(x.page_content for x in sparse_relations)

        rets = []
        pbar = tqdm(total=total, desc="Generating", ncols=100)
        for cidx, (ncnt, samples) in enumerate(zip(dist, csamples)):
            USER_PROMPT = gen_user_prompt(sparse_articles, relations, [G.nodes[n]["doc_enc"] for n in samples], input_post)
            cluster_rets = (
                [
                    {"role": "developer" if "gpt-4o" in self.llm_model else "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT},
                ]
                if self.llm_model != "gemma"
                else [{"role": "user", "content": SYSTEM_PROMPT + "\n" + USER_PROMPT}]
            )
            initial_res = run_prompt(
                self.llm_base_url,
                {
                    "model": self.llm_model,
                    "messages": cluster_rets,
                    "seed": np.random.randint(2**32),
                    "temperature": 1.2,
                    "max_completion_tokens": 100,
                    "extra_body": {"seed": np.random.randint(2**32)},
                },
            )
            rets.append(initial_res)
            cluster_rets.extend(
                [
                    {"role": "assistant", "content": strip_enclosing_quotes(initial_res["choices"][0]["message"]["content"])},
                    {"role": "user", "content": "As *ANOTHER* member of your community, post your reply/response DIRECTLY:"},
                ]
            )
            for reply_no in range(ncnt - 1):
                res = run_prompt(
                    self.llm_base_url,
                    {
                        "model": self.llm_model,
                        "messages": cluster_rets,
                        "seed": np.random.randint(2**32),
                        "temperature": 1.2,
                        "max_completion_tokens": 100,
                        "extra_body": {"seed": np.random.randint(2**32)},
                    },
                )
                cluster_rets.extend(
                    [
                        {"role": "assistant", "content": strip_enclosing_quotes(res["choices"][0]["message"]["content"])},
                        {"role": "user", "content": "As *ANOTHER* member of your community, post your reply/response DIRECTLY:"},
                    ]
                )
                rets.append(res)
                pbar.update()
        pbar.close()
        return [process_mentions(strip_enclosing_quotes(r["choices"][0]["message"]["content"])) for r in rets]


if __name__ == "__main__":
    G = pd.read_pickle("../data/graph/G.pkl")
    embs = pd.read_pickle("../data/intermediates/all_doc_nv2_inst_umap128.pkl")
    belief_embs = pd.read_pickle("../data/intermediates/all_data_belief_emb.pkl")
    emb_umapper = pd.read_pickle("../data/intermediates/all_doc_nv2_inst.umapper")
    sparse_embedder = SpladeEmbedding()
    # dense_embedder = Nv2Embedding(instruction="Represent the query for retrieving supporting documents.\n", device_map='cpu')

    dense_embedder = VoyageAIEmbeddings(
        model="voyage-3-large",
        api_key=os.getenv("VOYAGE_API_KEY"),
        batch_size=128,
    )

    scrag = SCRAG(G, embs, belief_embs, emb_umapper, "gpt-4o-mini", None)  # llm_model and llm_base_url
    scrag.init_vectorstore(sparse_embedder, dense_embedder, "news_splade", "news_kg", "docs_voyage_inst")
    scrag.predict("<Testing input here>", total=30)
