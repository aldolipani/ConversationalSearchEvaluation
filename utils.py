import re
import json
import numpy as np
from glob import glob
from collections import defaultdict
from numpy.random import choice
from itertools import product

class RegExp:
    
    def __init__(self):
        self.res = None
    
    def get(self, pattern, line):
        match = re.search(pattern, line)
        if match:
            self.res = match.group(1)
        else:
            self.res = None
        return self.res

#rel_kind = 2 # dataset relevance
rel_kind = 4 # user relevance
    
def get_topics_old():
    topics = set()
    for log in glob("./logs/*.log"):
        topics.add("_".join(log.split("/")[-1].split("_")[:2]))
    return sorted(list(topics))


def get_topics(path_topic):
    topics = {}
    with open(path_topic) as f:
        for line in f.readlines()[1:]:
            items = line.split('\t')
            topic_id = int(items[0].strip())
            topic = items[1].strip().lower()
            topics[topic_id] = topic
    return topics


def get_sub_topics(path_sub_topic):
    sub_topics = defaultdict(list)
    with open(path_sub_topic) as f:
        for line in f.readlines()[1:]:
            items = line.split('\t')
            sub_topic = int(items[0])
            topic = int(items[2])
            sub_topics[topic].append(sub_topic)
    return sub_topics

def get_num_question_answered(lines):
    reg_exp = RegExp()
    n_questions = 0
    n_questions_answered = 0
    trigger = False
    for line in lines:
        if reg_exp.get("(.+) \\| ", line):
            trigger = True
        elif reg_exp.get("satisfied\\? ([yYnN])", line):
            break
        elif trigger and reg_exp.get(".+\\? (.+)$", line):
            if reg_exp.res.strip():
                n_questions_answered += 1
            n_questions += 1
        else:
            trigger = False
    return n_questions_answered, n_questions

def get_num_question_answered_list(topic):
    res = []
    for log in glob("./logs/" + topic + "*.log"):
        with open(log) as f:
            item = get_num_question_answered(f.readlines())
            res.append(item)
    return res

# CBP Discount Function


def cbp(sequence, alpha = 0.8):
    res = 0.0
    for m, item in enumerate(sequence):
        if item[rel_kind]:
            res += dd_cbp(alpha, m)
    return res

def get_cbp_list(sequence_list, alpha = 0.8):
    cbp_scores = []
    for sequence in sequence_list:
        cbp_scores.append(cbp(sequence, alpha))
    return cbp_scores


def ncbp(sequence, alpha = 0.8):
    res = 0.0
    for m, item in enumerate(sequence):
        if item[rel_kind]:
            res += dd_ncbp(alpha, m)
        #print(res, dd_ncbp(alpha, m))
    return res

def get_ncbp_list(sequence_list, alpha = 0.8):
    ncbp_scores = []
    for sequence in sequence_list:
        ncbp_scores.append(ncbp(sequence, alpha))
    return ncbp_scores

# CBR+ Discount Function
def dd_cbp_p(alpha, beta, m, sequence):
    res = 1.0
    for item in sequence[:m]:
        if item[rel_kind]:
            res *= alpha
        else:
            res *= beta
    for _ in range(m - len(sequence)):
        res *= beta
    return res

def dd_cbp_p_new(alpha, beta, m, sequence):
    res = 1.0
    for item in sequence[:m]:
        if item[rel_kind]:
            beta += alpha
            beta = min(beta, 1.0)
        else:
            beta -= alpha
            beta = max(beta, 0.0)
        res *= beta
    for _ in range(m - len(sequence)):
        beta -= alpha
        beta = max(beta, 0.0)
        res *= beta
    return res

def cbp_p(sequence, alpha, beta):
    res = 0.0
    for m, item in enumerate(sequence):
        if item[rel_kind]:
            res += dd_cbp_p(alpha, beta, m, sequence)
    return res

def ncbp_p(sequence, alpha, beta):
    res = 0.0
    norm = 0.0
    for m, item in enumerate(sequence):
        val = dd_cbp_p(alpha, beta, m, sequence)
        if item[rel_kind]:
            res += val
        norm += val
    return res/norm

def get_cbp_tt_list(sequence_list, alpha, beta):
    cbp_tt_scores = []
    for sequence in sequence_list:
        cbp_tt_scores.append(cbp_tt(sequence, alpha, beta))
    return cbp_tt_scores


def get_ncbp_p_list(sequence_list, alpha, beta):
    res = []
    for sequence in sequence_list:
        res.append(ncbp_p(sequence, alpha, beta))
    return res

# Precision Discount Function


def ncp(sequence):
    res = 0.0
    for item in sequence:
        if item[rel_kind]:
            res += dd_cp(sequence)
    return res

def get_ncp_list(sequence_list):
    cp_scores = []
    for sequence in sequence_list:
        cp_scores.append(ncp(sequence))
    return cp_scores

# Satisfaction

def get_satisfaction(lines):
    reg_exp = RegExp()
    satisfaction = 0
    for line in lines:
        if reg_exp.get("satisfied\\? ([yYnN])", line):
            if reg_exp.res.lower() == "y":
                satisfaction = 1
    return satisfaction

# Transitions

def get_transitions(lines, include_rel = False):
    reg_exp = RegExp()
    n_subtopics = -1
    current_subtopic = 0
    subtopic = 0
    is_rel = False
    transitions = defaultdict(int)
    for line in lines:
        #print(line)
        if reg_exp.get("#sub-topics (\\d+)", line):
            n_subtopics = int(reg_exp.res)
        elif reg_exp.get("sub-topic (\\d+)", line):
            subtopic = int(reg_exp.res)
        elif reg_exp.get("relevant\\? ([yYnN])", line): # delayed
            if not include_rel:
                transitions[(current_subtopic, subtopic)] += 1
            else:
                transitions[(current_subtopic, subtopic, is_rel)] += 1
            is_rel = reg_exp.res.lower() == "y"
            current_subtopic = subtopic
    if not include_rel:
        transitions[(current_subtopic, n_subtopics + 1)] += 1
    else:
        transitions[(current_subtopic, n_subtopics + 1, is_rel)] += 1
    return transitions


def get_transitions_table(num_subtopics, transitions_list, epsilon = 0.0):
    transitions_table = defaultdict(float)
    for transitions in transitions_list:
        for key, count in transitions.items():
            transitions_table[key] += count
    
    norms = defaultdict(float)
    for from_subtopic, to_subtopic in product(range(num_subtopics+1), range(1, num_subtopics+2)):
        if not (from_subtopic == 0 and to_subtopic == num_subtopics + 1):            
            transitions_table[(from_subtopic, to_subtopic)] += epsilon
        norms[from_subtopic] += transitions_table[(from_subtopic, to_subtopic)]
    
    for from_subtopic, to_subtopic in product(range(num_subtopics+1), range(1, num_subtopics+2)):
        if norms[from_subtopic] > 0.0:
            transitions_table[(from_subtopic, to_subtopic)] /= norms[from_subtopic]
            
    return transitions_table


def get_transitions_tables(num_subtopics, transitions_list, epsilon = 0.0):
    rel_transitions_table = defaultdict(float)
    irr_transitions_table = defaultdict(float)
    for transitions in transitions_list:
        for key, count in transitions.items():
            if key[2]:
                rel_transitions_table[(key[0], key[1])] += count
            else:
                irr_transitions_table[(key[0], key[1])] += count
    
    rel_norms = defaultdict(float)
    for from_subtopic, to_subtopic in product(range(num_subtopics+1), range(1, num_subtopics+2)):
        if from_subtopic > 0:            
            rel_transitions_table[(from_subtopic, to_subtopic)] += epsilon
        rel_norms[from_subtopic] += rel_transitions_table[(from_subtopic, to_subtopic)]
    
    irr_norms = defaultdict(float)
    for from_subtopic, to_subtopic in product(range(num_subtopics+1), range(1, num_subtopics+2)):
        if not (from_subtopic == 0 and to_subtopic == num_subtopics + 1):
            irr_transitions_table[(from_subtopic, to_subtopic)] += epsilon
        irr_norms[from_subtopic] += irr_transitions_table[(from_subtopic, to_subtopic)]
    
    for from_subtopic, to_subtopic in product(range(num_subtopics+1), range(1, num_subtopics+2)):
        if rel_norms[from_subtopic] > 0.0:
            rel_transitions_table[(from_subtopic, to_subtopic)] /= (rel_norms[from_subtopic]) 
    
    for from_subtopic, to_subtopic in product(range(num_subtopics+1), range(1, num_subtopics+2)):
        irr_transitions_table[(from_subtopic, to_subtopic)] /= (irr_norms[from_subtopic])
    
    return rel_transitions_table, irr_transitions_table

def get_qrels(topic):
    # paragraph -> subtopic_set
    qrels = {}
    with open("t" + topic + "_prels.tsv") as file:
        for line in file.readlines():
            items = line.split("\t")
            if items[1]:
                qrels[int(items[0])] = set([int(subtopic_id) for subtopic_id in items[1].strip().split(" ")])
    return qrels

def get_sequence_old(lines, qrels):
    reg_exp = RegExp()
    current_subtopic = 0
    subtopic = 0
    paragraph = 0
    query = ""
    sequence = []
    for line in lines:
        if reg_exp.get("\\] (.+)  \\|", line):
            query = reg_exp.res.strip()
        elif reg_exp.get("picked par = (\\d+)", line):
            paragraph = int(reg_exp.res)
        elif reg_exp.get("sub-topic (\\d+)", line):
            subtopic = int(reg_exp.res)
        elif reg_exp.get("relevant\\? ([yYnN])", line):
            user_rel = reg_exp.res.lower() == "y"
            rel = False
            if paragraph in qrels and current_subtopic in qrels[paragraph]:
                rel = True
            sequence.append((current_subtopic, paragraph, rel, subtopic, user_rel, query))
            current_subtopic = subtopic
        
    return sequence

def get_session_ids(path_log):
    res = set()
    with open(path_log) as f:
        for line in f.readlines()[1:]:
            items = line.split('\t')
            user = items[3].strip()
            topic = int(items[4].strip())
            session_id = (user, topic)
            res.add(session_id)
    return list(res)

# new version
def get_sequence(lines):
    reg_exp = RegExp()
    current_sub_topic = 0
    subtopic = 0
    document = 0
    query = ""
    sequence = []
    for line in lines:
        #print(line)
        items = line.split('\t')
        content = items[2]
        if reg_exp.get(r'user queries subtopic (\d+)', content):
            sub_topic = int(reg_exp.res)
            reg_exp.get(r'with: (.+)', content)
            query = reg_exp.res.lower().strip()
            #print(sub_topic, query)
        elif reg_exp.get(r'engine returns: (\d+)', content):
            document = int(reg_exp.res)
            #print(document)
        elif reg_exp.get('users judges paragraph ' + str(document) + ' (.+)', content):
            rel = reg_exp.res
            #print(rel)
            action = (current_sub_topic, document, rel, sub_topic, query)
            if not (len(sequence) > 0 and action[4] == sequence[-1][4]): # remove duplicate interactions
                sequence.append(action)
            current_sub_topic = sub_topic
    return sequence

# new
def get_all_sequences(path_log, session_ids):
    res = defaultdict(list)
    for session_id in session_ids:
        selected_lines = []
        with open(path_log) as f:
            for line in f.readlines()[1:]:
                items = line.split('\t')
                user = items[3].strip()
                topic = int(items[4].strip())
                if user == session_id[0] and topic == session_id[1]:
                    selected_lines.append(line)

        sequence = get_sequence(selected_lines)
        res[session_id[1]].append(sequence)
    return res


def get_sequence_list(topic, qrels):
    sequence_list = []
    for log in glob("./logs/" + topic + "*.log"):
        with open(log) as f:
            lines = f.readlines()
            sequence = get_sequence(lines, qrels)
            sequence_list.append(sequence)
    return sequence_list


def get_transitions_list(topic, include_rel = False):
    transitions_list = []
    for log in glob("./logs/" + topic + "*.log"):
        with open(log) as f:
            transitions = get_transitions(f.readlines(), include_rel)
            transitions_list.append(transitions)
    return transitions_list

def get_satisfaction_list(topic):
    satisfaction_list = []
    for log in glob("./logs/" + topic + "*.log"):
        with open(log) as f:
            lines = f.readlines()
            satisfaction = get_satisfaction(lines)
            satisfaction_list.append(satisfaction)
    return satisfaction_list


# Parse SQUAD dataset

def parse_squad():
    with open("dev-v2.0.proc") as f:
        dataset = json.load(f)
    
    topics = {}
    documents = {}
    qrels = {}

    document_id = 1
    for topic in dataset["data"]:
        topic_id = topic["title"]
        topics[topic_id] = {}
        documents[topic_id] = {}
        qrels[topic_id] = {}
        for paragraph in topic["paragraphs"]:
            document = paragraph["context"]
            documents[topic_id][document_id] = document
            for qas in paragraph["qas"]:
                if not qas["is_impossible"]:
                    query_id = (document_id, qas["id"])
                    subtopic = qas["question"]
                    topics[topic_id][query_id] = subtopic
                    qrels[topic_id][(document_id, query_id)] = 1
            document_id += 1
    return topics, documents, qrels


def to_matrix(table):
    size = max([item[1] for item in table])
    matrix = []
    for i in range(0,size):
        matrix.append([])
        for j in range(size):
            if (i, j + 1) in table:
                matrix[i].append(table[(i, j + 1)])
            else:
                matrix[i].append(0.0)
    matrix = np.array(matrix)
    return matrix


def generate_sequence(num_subtopics):
    prob = 1.0
    sequence = []
    current_subtopic = 0
    
    next_subtopic = choice(range(1, num_subtopics + 1), 1)[0]
    prob *= 1.0/num_subtopics
    sequence.append((current_subtopic, next_subtopic))
    current_subtopic = next_subtopic
    
    while next_subtopic != num_subtopics + 1:
        next_subtopic = choice(range(1, num_subtopics + 2), 1)[0]
        prob *= 1.0/(num_subtopics + 1)
        sequence.append((current_subtopic, next_subtopic))
        current_subtopic = next_subtopic
        
    return sequence, prob

def generate_sequence_from_table(num_subtopics, table):
    prob = 1.0
    table = to_matrix(table)
    sequence = []
    current_subtopic = 0
    
    next_subtopic = choice(range(1, num_subtopics + 2), 1, p=table[0])[0]
    prob *= table[0, next_subtopic - 1]
    sequence.append((current_subtopic, next_subtopic))
    current_subtopic = next_subtopic
    
    while next_subtopic != num_subtopics + 1:
        next_subtopic = choice(range(1, num_subtopics + 2), 1, p=table[current_subtopic])[0]
        prob *= table[current_subtopic, next_subtopic-1]
        sequence.append((current_subtopic, next_subtopic))
        current_subtopic = next_subtopic
    
    return sequence, prob


class SequenceGenerator:
    
    def __init__(self, num_subtopics, rel_table, irr_table):
        self.num_subtopics = num_subtopics
        self.rel_table = to_matrix(rel_table)
        self.irr_table = to_matrix(irr_table)
        self.relevance = False
        self.current_subtopic = 0
        
    def __iter__(self):
        #self.current_subtopic = 0
        return self
    
    def set_relevance(self, relevance):
        self.relevance = relevance
    
    def __next__(self):
        next_subtopic = -1
        prob = -1
        if self.current_subtopic == self.num_subtopics + 1:
            raise StopIteration
        if self.relevance:
            next_subtopic = choice(range(1, self.num_subtopics + 2), 1, p=self.rel_table[self.current_subtopic])[0]
            prob = self.rel_table[self.current_subtopic, next_subtopic - 1]
        else:
            next_subtopic = choice(range(1, self.num_subtopics + 2), 1, p=self.irr_table[self.current_subtopic])[0]
            prob = self.irr_table[self.current_subtopic, next_subtopic - 1]
        self.current_subtopic = next_subtopic
        return next_subtopic, prob