# %%

from multiprocessing import Pool

import numpy as np
from tqdm import tqdm

from evaluate_standard_models import PoolHelper, get_all_evaluation_measures
from search_engine import *
from utils import get_session_ids, get_all_sequences, get_topics, get_sub_topics

np.set_printoptions(precision=3)


def main():
    path_log = './data/log.2.tsv'
    path_topic = './data/topic.tsv'
    path_sub_topic = './data/sub_topic.tsv'
    path_retrievable_paragraph = './data/retrievable_paragraph.tsv'

    # %%

    session_ids = get_session_ids(path_log)
    all_sequences = get_all_sequences(path_log, session_ids)

    # %%

    topics = get_topics(path_topic)

    # %%

    # topic -> sub_topic -> queries
    all_queries = {}

    for topic in all_sequences:

        queries = defaultdict(list)
        for sequence in all_sequences[topic]:
            for action in sequence:
                sub_topic = action[3]
                query = action[4]
                queries[sub_topic].append(query)

        all_queries[topic] = queries

    # %%

    # topic -> sub_topic -> paragraphs
    qrels = defaultdict(dict)

    sub_topic_to_paragraph = defaultdict(set)
    with open(path_retrievable_paragraph) as f:
        for line in f.readlines()[1:]:
            items = line.split('\t')
            sub_topic = int(items[0])
            paragraph = int(items[1])
            sub_topic_to_paragraph[sub_topic].add(paragraph)

    sub_topics = get_sub_topics(path_sub_topic)

    for topic in sub_topics:
        for sub_topic in sub_topics[topic]:
            qrels[topic][sub_topic] = sub_topic_to_paragraph[sub_topic]

    # %%

    # from users continuos
    # topic -> sub_topic -> paragraphs
    qrels_users = {}

    for topic in sub_topics:
        for sequence in all_sequences[topic]:
            for action in sequence:
                paragraph = action[1]
                rel = action[2]
                sub_topic = action[3]
                if topic not in qrels_users:
                    qrels_users[topic] = {}
                if sub_topic not in qrels_users[topic]:
                    qrels_users[topic][sub_topic] = {}
                if paragraph not in qrels_users[topic][sub_topic]:
                    qrels_users[topic][sub_topic][paragraph] = (0, 0)
                num, den = qrels_users[topic][sub_topic][paragraph]
                if rel == 'relevant':
                    qrels_users[topic][sub_topic][paragraph] = (num + 1, den + 1)
                else:
                    qrels_users[topic][sub_topic][paragraph] = (num, den + 1)

    for topic in qrels_users:
        for sub_topic in qrels_users[topic]:
            for paragraph in qrels_users[topic][sub_topic]:
                num, den = qrels_users[topic][sub_topic][paragraph]
                qrels_users[topic][sub_topic][paragraph] = num / den

    qrels_users

    # %% md

    # Setup Search System

    # %%

    direct_index = DirectIndex.load()

    # %%

    pre_preocessor = PreProcessor()

    print('#model: ', 'd-lm')

    max_necs = 0
    max_mu = 0
    for mu in tqdm(range(100 + 1)):
        mu = (mu + 1) * 10

        search = Search(None, {}, pre_preocessor, scorer=DirichletSmoothingLM(mu))
        pool_res = [PoolHelper(10000, 1, search, topics, sub_topics, qrels, qrels_users, direct_index, all_sequences,
                               all_queries)(0)]

        cur_necs = max(get_all_evaluation_measures(pool_res)['necs'])
        if max_necs < cur_necs:
            max_necs = cur_necs
            max_mu = mu

    # %% md

    print('#mu: ', max_mu)

    res = 10
    search = Search(None, {}, pre_preocessor, scorer=DirichletSmoothingLM(max_mu))
    noises = list(range(0, res + 1))
    pool = Pool(len(noises))
    max_pool_res = list(pool.imap_unordered(
        PoolHelper(10000, res, search, topics, sub_topics, qrels, qrels_users, direct_index, all_sequences,
                   all_queries), noises, chunksize=1))
    pool.terminate()

    all_evaluation_measures = get_all_evaluation_measures(max_pool_res)

    # %% md

    print('measure\tnoise\tscore')
    for m, l in all_evaluation_measures.items():
        for n, i in enumerate(l):
            print(m + '\t' + str(n / res) + '\t' + str(i))


if __name__ == '__main__':
    main()
