import argparse
import json
from pathlib import Path
from joblib import Parallel, delayed

import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
import contextlib
import joblib

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

SYSTEM_PROMPT = """I will provide you with a news article labeled as INPUT_NEWS. Your task is to extract structured information from it in the form of triplets for constructing knowledge graph. Each triplet should be in the form of (h:type, r, o:type), where 'h' stands for the head entity, 'r' for the relationship, and 'o' for the tail entity. The 'type' denotes the category of the corresponding entity.

---

### Predefined Entities:
The Entities should be non-generic and can be classified into the following categories:
ORG: Organizations other than government or regulatory bodies (e.g., "Apple Inc.")
ORG/GOV: Government bodies (e.g., "United States Government")
ORG/REG: Regulatory bodies (e.g., "Federal Reserve")
COMP: Companies (e.g., "Google")
PERSON: Individuals (e.g., "Elon Musk", "Queen Elizabeth II")
GPE: Geopolitical entities such as countries, cities, etc. (e.g., "Ukraine", "Gaza")
GEO: Geographical locations relevant to conflicts or politics (e.g., "Donbas Region", "West Bank")
PRODUCT: Products or services (e.g., "Bayraktar TB2 Drone")
SECTOR: Company sectors or industries (e.g., "Defense industry")
ECON_INDICATOR: Economic indicators (e.g., "Inflation rate")
ARMY: Military forces, paramilitary groups, militias, or peacekeeping forces (e.g., "Ukrainian Armed Forces", "IDF", "Wagner Group")
RESISTANCE_GROUP: Rebel groups, insurgent organizations, or terrorist groups (e.g., "Hamas", "ISIS", "Taliban")
POL_PARTY: Political parties (e.g., "Democratic Party", "United Russia")
ADV_GROUP: Advocacy groups or non-governmental organizations (e.g., "Amnesty International", "Red Cross")
LEGISLATION: Laws, policies, or regulations (e.g., "Martial Law in Ukraine")
POL_ISSUE: Political issues or topics (e.g., "NATO Expansion", "Two-State Solution")
POL_EVENT: Political events (e.g., "Presidential Debate", "Impeachment Trial")
EVENT: Specific and material events (e.g., "Russia-Ukraine War", "Hamas Attack on Israel")
CRISIS: Large-scale crises or global challenges (e.g., "COVID-19 Pandemic", "2008 Financial Crisis")
MEDIA: News agencies, media outlets, or platforms (e.g., "BBC", "Al Jazeera")
TREATY: International treaties or agreements (e.g., "Minsk Agreement", "Oslo Accords")


---

### Predefined Relationships:
The Relationships r between these entities must be represented by one of the following verbs:
- Has, Announces, Introduces, Produces, Controls, Operates_In, Participates_In, Relates_To, Impacts, Supports, Opposes, Advocates_For, Proposes, Passes_Law, Challenges, Negotiates_With, Signs_Treaty_With, Legislates, Vetoes, Amends, Ratifies, Appoints, Invests_In, Raises_Funds, Decreases, Acquires, Merges_With, Divests_From, Sells, Buys, Lends_To, Borrows_From, Grants, Subsidizes, Outsources, Conflicts_With, Sanctions, Mediates, Occupies, Attacks, Cheats, Invades, Withdraws_From, Deploys_Troops_To, Protests_Against, Campaigns_For, Mobilizes, Retreats_From, Defends, Forms_Alliance_With, Severs_Ties_With, Embargoes, Endorses, Criticizes, Coordinates_With, Collaborates_With, Wins_Election, Loses_Election, Dies.

---

### Instructions:
- Remember to conduct entity disambiguation, consolidating different phrases or acronyms that refer to the same entity (for instance, "UK Central Bank", "BOE" and "Bank of England" should be unified as "Bank of England"). 
- Simplify each entity of the triplet to be no more than three words.
- Your output should strictly consist of a list of triplets and nothing else. Do NOT include redundant triplets or triplets with numerical or date entities.

"""
    
USER_PROMPT = """
### Example:
Consider the following news excerpt:  
"The Democratic Party announced its candidate for the upcoming elections, while opposition groups staged protests against recent legislation."

From this text, your output should be an array of triplets as follows:
[
{"Democratic Party": "POL_PARTY", "Announces": "r", "Candidate": "PERSON"},
{"Opposition Groups": "ADV_GROUP", "Protests_Against": "r", "Recent Legislation": "LEGISLATION"},
{"Upcoming Elections": "POL_EVENT", "Relates_To": "r", "Democratic Party": "POL_PARTY"}
]

---

Now, let's apply this process to the following INPUT_NEWS:

"""



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, required=True, help='specify input jsonl')
    parser.add_argument('--joblib', type=int, default=1, help='use joblib (1 means no parallelism)')
    args = parser.parse_args()
    args.input = Path(args.input).resolve()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096 * 2, chunk_overlap=0)
    chunks = []
    with open(args.input, 'r', encoding='utf-8') as f:
        for l in f:
            tmp = json.loads(l)
            splits = text_splitter.split_text(tmp['content'])
            for j, doc in enumerate(splits):
                new_record = dict(**tmp)
                new_record["content"] = doc
                new_record["chunk_seq"] = j
                new_record["total_chunk"] = len(splits)
                chunks.append(new_record)
    print("Loaded", len(chunks), "documents")

    results = []
    # Process each document (two attempts)
    def process_doc(doc):
        @retry(wait=wait_random_exponential(multiplier=1, max=8), stop=stop_after_attempt(4))
        def completion_with_backoff():
            client = openai.Client(
                api_key="your-api-key-here",  # Replace with your actual API key
                base_url="http://128.174.212.200:11435/v1",
            )
            completion = client.chat.completions.create(
                model="llama",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT + doc['content']}
                ],
                temperature=0.0,
                response_format={ "type": "json_object" }
            )
            try:
                triplets = json.loads(completion.choices[0].message.content)
            except Exception:
                print("error", doc['url'], doc['chunk_seq'])
                triplets = []
            return triplets
        
        triplets = completion_with_backoff()
        return {"url": doc['url'], "chunk_seq": doc['chunk_seq'], "triplets": triplets}
    if args.joblib == 1:
        for doc in tqdm(chunks, ncols=100):
            results.append(process_doc(doc))
    else:
        with tqdm_joblib(tqdm(chunks, ncols=100)):
            results = Parallel(n_jobs=args.joblib)(delayed(process_doc)(doc) for doc in chunks)
        
    Path('../../data/kg').mkdir(parents=True, exist_ok=True)
    with open("../../data/kg/chunks.jsonl", "w", encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result) + "\n")
    print("Done")
    

if __name__ == "__main__":
    main()