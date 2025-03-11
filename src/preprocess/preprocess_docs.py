
#%%
import sys
from pathlib import Path  # if you haven't already done so

file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

# Additionally remove the current file's directory from sys.path
try:
    sys.path.remove(str(parent))
except ValueError: # Already removed
    pass
import html
import json
import re
from collections import Counter, deque
from copy import deepcopy
from pathlib import Path

import networkx as nx
import pandas as pd
import requests
from tqdm import tqdm
from utils.clean_text import clean_word_bag

#%%
stopwords = list(map(str.strip, Path('../utils/stopwords_en.txt').read_text().strip().split('\n')))
#%%
def extract_tco_urls(tweet_text):
    pattern = r'https?://t\.co/\w+'
    links = re.findall(pattern, tweet_text)
    return links
# %%
print('Loading data')
all_data = {}
with open('../../data/social_media/tweets.jsonl', 'r', encoding='utf-8') as f:
    for l in f:
        tmp = json.loads(l)
        all_data[tmp['id']] = tmp
# df = pd.DataFrame.from_dict(all_data, orient='index').reset_index(drop=True)
# %%
print('Filtering reposts')
all_data_retweet = {}
for k, v in all_data.items():
    if tuple(sorted([x['type'] for x in v['referenced_tweets']])) == ('retweeted',):
        all_data_retweet[k] = v
print(len(all_data_retweet))
for k in all_data_retweet:
    all_data.pop(k)
# %%
cnt = 0
for k, v in all_data.items():
    if len(v['referenced_tweets']) == 1 and v['referenced_tweets'][0]['type'] == 'quoted':
        v['referenced_tweets'][0]['type'] = 'replied_by_quote'
        cnt += 1
print(cnt)
#%%
print('Calculating cleaned word counts')
for k, v in tqdm(all_data.items(), ncols=100):
    bag = clean_word_bag(v['text'])
    v['clean_word_bag'] = bag
    v['clean_word_count'] = len(bag)
# %%
def expand_url(row):
    if len(row['entities']['urls']) > 0:
        urls = []
        existed_media = set()
        last_end = -1
        for u in deepcopy(row['entities']['urls']):
            if ('start' not in u or 'end' not in u) and 'indices' not in u:
                raise Exception(u)
            if 'start' not in u or 'end' not in u:
                u['start'] = u['indices'][0]
                u['end'] = u['indices'][1]
            if u['start'] < last_end:
                continue
            last_end = u['end']
            if 'media_key' in u and u['media_key'] is not None and u['media_key'] != '':
                existed_media.add(u['expanded_url'])
                u['expanded_url'] = '<MEDIA_ITEM>'
            urls.append(u)
        if any(x['type'] == 'quoted' for x in row['referenced_tweets']):
            quote_id = None
            for x in row['referenced_tweets']:
                if x['type'] == 'quoted':
                    quote_id = x['id']
                    break
            i = len(urls) - 1
            while i >= 0:
                if quote_id in urls[i]['expanded_url']:
                    urls[i]['expanded_url'] = '<QUOTE>...</QUOTE>'
                    break
                i -= 1
            while i >= 0:
                if quote_id in urls[i]['expanded_url']:
                    urls[i]['expanded_url'] = ''
                i -= 1
        if any(x['type'] == 'replied_by_quote' for x in row['referenced_tweets']):
            quote_id = None
            for x in row['referenced_tweets']:
                if x['type'] == 'replied_by_quote':
                    quote_id = x['id']
                    break
            i = len(urls) - 1
            while i >= 0:
                if quote_id in urls[i]['expanded_url']:
                    urls[i]['expanded_url'] = ''
                i -= 1
        for u in urls:
            if ('expanded_url' not in u and 't.co' in u['display_url']) or (('twitter.com' in u['expanded_url'] or 'x.com' in u['expanded_url']) and '/status/' in u['expanded_url']):
                u['expanded_url'] = '<ANOTHER_TWEET>'
        urls = sorted(urls, key=lambda x: x['start'])
        for i in range(1, len(urls)):
            if urls[i-1]['end'] > urls[i]['start']:
                print(row)
            assert urls[i-1]['end'] <= urls[i]['start']
        for url in urls:
            if row['text'][url['start']:url['end']] != url['url']:
                print(row)
            assert row['text'][url['start']:url['end']] == url['url']
        cursor = 0
        new_text = []
        for url in urls:
            new_text.append(row['text'][cursor:url['start']])
            new_text.append(url['expanded_url'])
            cursor = url['end']
        new_text.append(row['text'][cursor:])
        return html.unescape(requests.utils.unquote(''.join(new_text))).strip()
    else:
        return html.unescape(requests.utils.unquote(row['text'])).strip()
def process_mentions(tweet, keep_n=2):
    def replace_mention_group(match):
        mentions = re.findall(r'@\w+', match.group())
        num_mentions = len(mentions)
        
        if keep_n == 0:
            replace_count = num_mentions
        else:
            if num_mentions <= keep_n:
                return match.group()  # No replacement needed
            replace_count = num_mentions - keep_n
        
        # Determine replacement string
        if replace_count == 1:
            replacement = '@<USER_MENTION>'
        else:
            replacement = '@<USER_MENTION_LIST>'
        
        if keep_n == 0:
            return replacement
        else:
            # Retain the first 'keep_n' mentions and append replacement
            retained = ' '.join(mentions[:keep_n])
            return f'{retained} {replacement}'
    
    # Regex matches any group of mentions (1 or more)
    processed_tweet = re.sub(r'@\w+(?:\s+@\w+)*', replace_mention_group, tweet)
    return processed_tweet
# %%
print('Processing texts')
for k in all_data:
    all_data[k]['decoded_text'] = process_mentions(expand_url(all_data[k]))
#%%
for k, v in all_data.items():
    if len(v['referenced_tweets']) > 0 and any(x['type'] == 'quoted' for x in v['referenced_tweets']):
        if '<QUOTE>...</QUOTE>' not in v['decoded_text']:
            urls = extract_tco_urls(v['decoded_text'])
            if len(urls) > 0:
                v['decoded_text'] = v['decoded_text'].replace(urls[-1], '<QUOTE>...</QUOTE>')
            else:
                v['decoded_text'] = v['decoded_text'] + '<QUOTE>...</QUOTE>'
#%%
for k, v in all_data.items():
    if len(v['referenced_tweets']) > 0 and any(x['type'] == 'quoted' for x in v['referenced_tweets']):
        quote_id = None
        for x in v['referenced_tweets']:
            if x['type'] == 'quoted':
                quote_id = x['id']
                break
        if quote_id in all_data:
            v['expanded_text'] = v['decoded_text'].replace('<QUOTE>...</QUOTE>', '\n<QUOTE>\n' + all_data[quote_id]['decoded_text'] + '\n</QUOTE>\n')
for k, v in all_data.items():
    if 'expanded_text' not in v:
        all_data[k]['expanded_text'] = v['decoded_text']
# %%
cnt = 0
for k, v in all_data.items():
    if len(v['referenced_tweets']) == 2:
        if v['referenced_tweets'][0]['type'] == 'quoted':
            v['referenced_tweets'][1]['type'] = 'replied_with_quote'
            v['referenced_tweets'].pop(0)
        elif v['referenced_tweets'][1]['type'] == 'quoted':
            v['referenced_tweets'][0]['type'] = 'replied_with_quote'
            v['referenced_tweets'].pop(1)
        cnt += 1
print(cnt)
# %%
print('Consolidating same author message chains')
consolidG = nx.MultiDiGraph()
for row in tqdm(all_data.values(), ncols=100):
    consolidG.add_node(
        row['id'],
        kind='post',
        text=row['expanded_text'],
        text_chain=None,
    )
for row in tqdm(all_data.values(), ncols=100):
    if len(row['referenced_tweets']) > 0:
        ref = row['referenced_tweets'][0]
        if ref['id'] in all_data and all_data[ref['id']]['author_id'] == row['author_id']:
            consolidG.add_edge(ref['id'], row['id'], key='reply', kind='reply', timestamp=row['created_at'])
consolidG.remove_nodes_from(list(nx.isolates(consolidG)))
print(consolidG)
# %%
for i, component in enumerate(nx.weakly_connected_components(consolidG)):
    subgraph = consolidG.subgraph(component)
    source_nodes = [n for n in subgraph if subgraph.in_degree(n) == 0]
    assert len(source_nodes) == 1
    node = sorted(source_nodes)[0]
    queue = deque([(node, [])])
    while len(queue) > 0:
        node, cur_arr = queue.popleft()
        consolidG.nodes[node]['text_chain'] = deepcopy(cur_arr[-4:])
        next_arr = cur_arr + [consolidG.nodes[node]['text']]
        for succ in consolidG.successors(node):
            queue.append((succ, next_arr))
# %%
for n, d in consolidG.nodes(data=True):
    if d['text_chain']:
        all_data[n]['final_text'] = '\n\n'.join(d['text_chain']) + '\n\n' + all_data[n]['expanded_text']
    else:
        all_data[n]['final_text'] = all_data[n]['expanded_text']
for k, v in all_data.items():
    if 'final_text' not in v:
        all_data[k]['final_text'] = v['expanded_text']
for k in all_data:
    all_data[k]['final_text'] = all_data[k]['final_text'].strip()
# %%
print('Generating engagement graph')
G = nx.MultiDiGraph()
for row in tqdm(all_data.values(), ncols=100):
    G.add_node(
        row['id'],
        kind='post',
        author_id=row['author_id'],
        created_at=row['created_at'],
        conversation_id=row['conversation_id'],
        text=row['final_text'].strip(),
        lang=row['lang'],
        clean_word_bag=row['clean_word_bag'],
        clean_word_count=row['clean_word_count'],
        data_source=row['data_source'],
    )
for row in tqdm(all_data.values(), ncols=100):
    for ref in row['referenced_tweets']:
        if ref['id'] in all_data:
            G.add_edge(ref['id'], row['id'], key='replied_by', kind='replied_by', timestamp=row['created_at'])
# %%
to_be_removed = []
for n in consolidG.nodes():
    succs = list(G.successors(n))
    authors = set([G.nodes[succ]['author_id'] for succ in succs])
    if len(authors) == 1 and G.nodes[n]['author_id'] in authors:
        to_be_removed.append(n)
print(len(to_be_removed))
G.remove_nodes_from(to_be_removed)
G.remove_nodes_from(list(nx.isolates(G)))
print(G)
#%%
to_be_pruned = []
unacceptable_lang = {'qme', 'zxx', 'qam', 'qht', 'art', 'qst', 'ckb', 'qct'}
for n, d in G.nodes(data=True):
    if len(d['text'].split()) < 5 or len(d['text'].split()) > 260 or d['lang'] in unacceptable_lang or d['clean_word_count'] < 4 or d['clean_word_count'] > 260 or d['lang'] in unacceptable_lang:
        to_be_pruned.append(n)
# print(len(to_be_pruned))
G.remove_nodes_from(to_be_pruned)
G.remove_nodes_from(list(nx.isolates(G)))
print(G)
# %%
good_node_set = set(G.nodes())
G.remove_edges_from(list(G.edges()))
for row in tqdm(all_data.values(), ncols=100):
    for ref in row['referenced_tweets']:
        if ref['id'] in good_node_set and row['id'] in good_node_set:
            G.add_edge(row['id'], ref['id'], key='reply', kind='reply', timestamp=row['created_at'])
print(G)
# %%
def trace_post_history(graph, start_node):
    history = []
    queue = deque([start_node])
    while True:
        cur_node = queue.popleft()
        text = graph.nodes[cur_node]["text"]
        history.append((text,))
        succs = list(graph.successors(cur_node))
        assert len(succs) <= 1
        if len(succs) == 0:
            break
        queue.append(succs[0])
    return history
def gen_doc(history):
    history = history[::-1]
    if len(history) == 1:
        doc_enc = history[0][0]
    elif len(history) == 2:
        doc_enc = "The context for the response:\n<CONTEXT>\n" + history[0][0] + "\n</CONTEXT>\n\nRESPONSE:\n" + history[1][0]
    elif len(history) == 3:
        doc_enc = "A thread of posts on Twitter as the context for the response:\n<CONTEXT>\n" + history[0][0] + "\n----------\n" + \
            history[1][0] + "\n</CONTEXT>\n\nRESPONSE:\n" + history[2][0]
    elif len(history) == 4:
        doc_enc = "A thread of posts on Twitter as the context for the response:\n<CONTEXT>\n" + history[0][0] + "\n----------\n" + \
            "\n----------\n".join(map(lambda x: x[0], history[-3:-1])) + "\n</CONTEXT>\n\nRESPONSE:\n" + history[-1][0]
    elif len(history) >= 5:
        doc_enc = "A thread of posts on Twitter as the context for the response:\n<CONTEXT>\n" + history[0][0] + "\n----------\n" + \
            ("... (omitted)\n----------\n" if len(history) > 4 else "") + \
            "\n----------\n".join(map(lambda x: x[0], history[-4:-1])) + "\n</CONTEXT>\n\nRESPONSE:\n" + history[-1][0]
    return doc_enc.strip()
# %%
for n, d in tqdm(G.nodes(data=True), ncols=100):
    history = trace_post_history(G, n)
    d['cascade_depth'] = len(history)
    d['engagement_in_degree'] = G.in_degree(n)
    d['doc_enc'] = gen_doc(history)
#%%
to_be_pruned2 = []
for n, d in G.nodes(data=True):
    if len(d['doc_enc'].split()) > 630 or len(d['doc_enc']) > 4000:
        to_be_pruned2.append(n)
G.remove_nodes_from(to_be_pruned2)
G.remove_nodes_from(list(nx.isolates(G)))
print(G)
# %%
for n, d in tqdm(G.nodes(data=True), ncols=100):
    history = trace_post_history(G, n)
    d['cascade_depth'] = len(history)
    d['engagement_in_degree'] = G.in_degree(n)
    d['doc_enc'] = gen_doc(history)
# %%
print('Saving graph')
Path('../../data/graph').mkdir(parents=True, exist_ok=True)
pd.to_pickle(G, '../../data/graph/G.pkl')
#%%
data_sources = []
for n, d in G.nodes(data=True):
    data_sources.extend(list(d['data_source']))
data_sources = dict(Counter(data_sources).most_common())
for k, v in data_sources.items():
    print(f'{k}: {v} documents')
# %%
print('Generating documents')
doc_db = set()
for n, d in tqdm(G.nodes(data=True), ncols=100):
    doc_db.add(d['doc_enc'])
doc_db = sorted(doc_db)
print(len(doc_db), 'documents')
# %%
print('Preparing for embedding step')
j = 0
Path('../../data/social_media_docs').mkdir(exist_ok=True)
for i in range(0, len(doc_db), 32*128):
    with open(f'../../data/social_media_docs/doc_set_{j}.pkl', 'wb') as f:
        pd.to_pickle(doc_db[i:i+32*128], f)
        j += 1
# %%
