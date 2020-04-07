# %%

import numpy as np
from utils import get_session_ids, get_all_sequences, get_topics, get_sub_topics, product, to_matrix
from search_engine import *
from numpy.random import choice
from tqdm.notebook import tqdm
from multiprocessing import Pool

np.set_printoptions(precision=3)

# %% md

# Data Preprocessing

# %%

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


def select_random_documents(documents: set, p: float = 0.0):
    assert (0.0 <= p <= 1.0)
    if p == 0.0:
        return set()
    elif p == 1.0:
        return documents
    else:
        return set(choice(list(documents), replace=False, size=int(len(documents) * p)))


def get_search_engine(topic, sub_topic, qrels, direct_index, noise=0.0):
    # select documents belonging to the topic
    selected_documents = {}
    for document in qrels[topic][sub_topic]:
        selected_documents[document] = direct_index.index[document]

    # select random documents
    # 1. from topic documents
    topic_documents = set()
    for _, sub_topic_documents in qrels[topic].items():
        topic_documents.update(sub_topic_documents)
    topic_documents = select_random_documents(topic_documents, noise)

    for document in topic_documents:
        selected_documents[document] = direct_index.index[document]

    # 2. from all_documents
    all_documents = direct_index.index.keys()
    all_documents = select_random_documents(all_documents, noise)

    for document in all_documents:
        selected_documents[document] = direct_index.index[document]

    # retrievable documents
    inverted_index = InvertedIndex()
    inverted_index.create(selected_documents)

    return selected_documents, inverted_index


# %%

def get_transitions_tables(num_subtopics, min_sub_topic, sequences, epsilon=0.0):
    rel_transitions_table = defaultdict(float)
    irr_transitions_table = defaultdict(float)
    p_rel = defaultdict(float)
    p_irr = defaultdict(float)
    for sequence in sequences:
        last_key = None
        last_rel = 'irrelevant'
        for n, action in enumerate(sequence):
            if n == 0:
                key = (action[0], action[3] - min_sub_topic + 1)
            else:
                key = (action[0] - min_sub_topic + 1, action[3] - min_sub_topic + 1)
            if last_rel == 'relevant':
                rel_transitions_table[key] += 1
                p_rel[key[0]] += 1
            else:
                irr_transitions_table[key] += 1
                p_irr[key[0]] += 1
            last_key = key
            last_rel = action[2]

        if last_rel == 'relevant':
            rel_transitions_table[(last_key[1], num_subtopics + 1)] += 1
            p_rel[last_key[1]] += 1
        else:
            irr_transitions_table[(last_key[1], num_subtopics + 1)] += 1
            p_irr[last_key[1]] += 1

    rel_norms = defaultdict(float)
    for from_subtopic, to_subtopic in product(range(num_subtopics + 1), range(1, num_subtopics + 2)):
        if from_subtopic > 0:
            rel_transitions_table[(from_subtopic, to_subtopic)] += epsilon
        rel_norms[from_subtopic] += rel_transitions_table[(from_subtopic, to_subtopic)]

    irr_norms = defaultdict(float)
    for from_subtopic, to_subtopic in product(range(num_subtopics + 1), range(1, num_subtopics + 2)):
        if not (from_subtopic == 0 and to_subtopic == num_subtopics + 1):
            irr_transitions_table[(from_subtopic, to_subtopic)] += epsilon
        irr_norms[from_subtopic] += irr_transitions_table[(from_subtopic, to_subtopic)]

    for from_subtopic, to_subtopic in product(range(num_subtopics + 1), range(1, num_subtopics + 2)):
        if rel_norms[from_subtopic] > 0.0:
            rel_transitions_table[(from_subtopic, to_subtopic)] /= (rel_norms[from_subtopic])

    for from_subtopic, to_subtopic in product(range(num_subtopics + 1), range(1, num_subtopics + 2)):
        if irr_norms[from_subtopic] > 0.0:
            irr_transitions_table[(from_subtopic, to_subtopic)] /= (irr_norms[from_subtopic])

    p = []
    for t in range(num_subtopics + 1):
        if (p_rel[t] + p_irr[t]) > 0:
            p.append(p_rel[t] / (p_rel[t] + p_irr[t]))
        else:
            p.append(0.0)
    p = np.array([p])

    return rel_transitions_table, irr_transitions_table, p, 1 - p


# %%

class SequenceGenerator:

    def __init__(self, num_subtopics, min_sub_topic, rel_table, irr_table):
        self.num_subtopics = num_subtopics
        self.rel_table = to_matrix(rel_table)
        self.irr_table = to_matrix(irr_table)
        self.min_sub_topic = min_sub_topic
        self.relevance = False
        self.current_subtopic = 0

    def __iter__(self):
        # self.current_subtopic = 0
        return self

    def set_relevance(self, relevance):
        self.relevance = relevance

    def __next__(self):
        next_subtopic = -1
        prob = -1
        if self.current_subtopic + self.min_sub_topic - 1 == self.num_subtopics + 1 + self.min_sub_topic - 1:
            raise StopIteration
        if self.relevance:
            # print(self.rel_table[self.current_subtopic].sum())
            next_subtopic = choice(range(1, self.num_subtopics + 2),
                                   1,
                                   p=self.rel_table[self.current_subtopic])[0]
            prob = self.rel_table[self.current_subtopic, next_subtopic - 1]
        else:
            # print(self.irr_table[self.current_subtopic].sum())
            next_subtopic = choice(range(1, self.num_subtopics + 2),
                                   1,
                                   p=self.irr_table[self.current_subtopic])[0]
            prob = self.irr_table[self.current_subtopic, next_subtopic - 1]
        self.current_subtopic = next_subtopic
        return next_subtopic + self.min_sub_topic - 1, prob


# %% md

# Correlation Analysis with Satisfaction

# %%

# CP

def dd_cp(alpha, m, l):
    return 1


def ncp(sequence):
    res = 0.0
    for action in sequence:
        if action[2] == 'relevant':
            res += dd_cp(None, None, None)
    return res / len(sequence)


# CBP

def dd_cbp(alpha, m, l=None):
    return alpha ** m


def dd_ncbp(alpha, m, l):
    return dd_cbp(alpha, m, l)


def ncbp(sequence, alpha):
    norm = 0.0
    for i in range(len(sequence)):
        norm += dd_cbp(alpha, i, len(sequence))
    res = 0.0
    for m, action in enumerate(sequence):
        if action[2] == 'relevant':
            res += dd_ncbp(alpha, m, len(sequence))
    return res / norm


# ECS

def dd_necs(alpha_rel, alpha_irr, m, sequence):
    res = 1.0
    for action in sequence[:m]:
        if action[2] == 'relevant':
            res *= alpha_rel
        else:
            res *= alpha_irr
    return res


def necs(sequence, alpha_rel, alpha_irr):
    norm = 0.0
    for i in range(len(sequence)):
        norm += dd_cbp(alpha_rel, i)
    res = 0.0
    for m, action in enumerate(sequence):
        if action[2] == 'relevant':
            res += dd_necs(alpha_rel, alpha_irr, m, sequence)
    return res / norm


# %% md

# Noise Analysis

# %%

num_samples = 1
res = 10


# %%

class PoolHelper(object):
    def __init__(self, num_samples, res, search):
        self.num_samples = num_samples
        self.res = res
        self.search = search

    def __call__(self, noise):
        return get_scores_with_noise(noise, self.num_samples, self.res, self.search)


def get_scores_with_noise(noise, num_samples, res, search):
    noise = noise / res

    topic_scores_necs = []
    topic_scores_ncbp = []
    topic_scores_ncp = []
    for topic in topics:
        num_subtopics = len(sub_topics[topic])

        all_r_documents = {}
        all_r_inverted_index = {}
        for sub_topic in sub_topics[topic]:
            r_documents, r_inverted_index = get_search_engine(topic, sub_topic, qrels, direct_index, noise)
            all_r_documents[sub_topic] = r_documents
            all_r_inverted_index[sub_topic] = r_inverted_index

        rel_table, irr_table, p_rel, _ = get_transitions_tables(num_subtopics,
                                                                min(sub_topics[topic]),
                                                                all_sequences[topic],
                                                                10e-6)
        scores_necs = []
        scores_ncbp = []
        scores_ncp = []
        for _ in range(num_samples):
            sg = SequenceGenerator(num_subtopics, min(sub_topics[topic]), rel_table, irr_table)
            prob = 1.0
            prev_sub_topic = 0
            sequence = []
            for m, (sub_topic, prob_action) in enumerate(sg):

                if sub_topic != num_subtopics + 1 + min(sub_topics[topic]) - 1:
                    query = ''
                    if sub_topic in all_queries[topic] and len(all_queries[topic][sub_topic]) > 0:
                        queries = all_queries[topic][sub_topic]
                        query = queries[choice(range(len(queries)))]

                    search.set_indices(all_r_inverted_index[sub_topic], all_r_documents[sub_topic])
                    answer = search.search(query,
                                           n=1,
                                           retrievable_paragraphs=all_r_documents[sub_topic].keys())[0]

                    if sub_topic in qrels_users[topic] and answer in qrels_users[topic][sub_topic] and \
                            np.random.uniform(0, 1) < qrels_users[topic][sub_topic][answer]:
                        sg.set_relevance(True)
                        sequence.append((prev_sub_topic, answer, 'relevant', sub_topic, query))
                    else:
                        sg.set_relevance(False)
                        sequence.append((prev_sub_topic, answer, 'irrelevant', sub_topic, query))

                prev_sub_topic = sub_topic
                prob *= prob_action

            scores_necs.append(necs(sequence, 0.85, 0.64))
            scores_ncbp.append(ncbp(sequence, 0.79))
            scores_ncp.append(ncp(sequence))

        scores_necs = np.array(scores_necs)
        topic_score_necs = np.mean(scores_necs)
        topic_scores_necs.append((topic, topic_score_necs))

        scores_ncbp = np.array(scores_ncbp)
        topic_score_ncbp = np.mean(scores_ncbp)
        topic_scores_ncbp.append((topic, topic_score_ncbp))

        scores_ncp = np.array(scores_ncp)
        topic_score_ncp = np.mean(scores_ncp)
        topic_scores_ncp.append((topic, topic_score_ncp))

    return noise, topic_scores_necs, topic_scores_ncbp, topic_scores_ncp


# %%

def get_all_evaluation_measures(pool_res):
    ys_ncp = []
    ys_ncbp = []
    ys_necs = []
    for noise, topic_scores_necs, topic_scores_ncbp, topic_scores_ncp in pool_res:
        ys_necs.append(np.mean([tv[1] for tv in topic_scores_necs]))
        ys_ncbp.append(np.mean([tv[1] for tv in topic_scores_ncbp]))
        ys_ncp.append(np.mean([tv[1] for tv in topic_scores_ncp]))

    return {'necs': ys_necs, 'ncbp': ys_ncbp, 'ncp': ys_ncp}


# %% md

## TF-IDF

# %%

res = 10

search = Search(inverted_index=None, documents={}, pre_processor=pre_preocessor, scorer=TFIDF())

noises = list(range(0, res+1))
pool = Pool(len(noises))
pool_res = list(pool.imap_unordered(PoolHelper(10000, res, search), noises, chunksize=1))
pool.terminate()

tfidf_pool_res = pool_res
all_evaluation_measures = get_all_evaluation_measures(pool_res)

# %% md

print('#model: ', 'tf-idf')
print('measure\tnoise\tscore')
for m, l in all_evaluation_measures.items():
    for n, i in enumerate(l):
        print(m + '\t' + str(n/res) + '\t' + str(i))
