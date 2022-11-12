import math
import pickle
from collections import defaultdict
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')
stop_words = set(stopwords.words('english'))


def preprocess(text):
    bag_of_words = defaultdict(int)
    tokens = tokenizer.tokenize(text)
    for token in tokens:
        token = token.lower()
        if token not in stop_words:
            bag_of_words[token] += 1
    return bag_of_words


def get_direct_index(documents):
    direct_index = {}
    for topic_id in documents:
        for document_id in documents[topic_id]:
            document = documents[topic_id][document_id]
            direct_index[(topic_id, document_id)] = preprocess(document)
    return direct_index


def get_inverted_index(direct_index):
    dictionary = defaultdict(int)
    cl = 0
    lengths = defaultdict(int)
    inverted_index = defaultdict(list)
    for document_id in direct_index:
        document = direct_index[document_id]
        for term in document:
            tf = direct_index[document_id][term]
            inverted_index[term] += [(document_id, tf)]
            cl += tf
            dictionary[term] += tf
            lengths[document_id] += tf
    return dictionary, cl, lengths, inverted_index


def language_model(tf, cf, cl, ld, mu=100):
    return (tf + mu * cf / cl) / (mu + ld);


def search(query, dictionary, cl, lengths, inverted_index, mu=100, n=1):
    query_bag_of_words = preprocess(query)
    query_bag_of_words = [item for item in query_bag_of_words.items() if item[0] in dictionary]
    results = defaultdict(float)
    results[(None, None)] = 0.0
    for term in query_bag_of_words:
        posting_list = inverted_index[term[0]]
        for post in posting_list:
            tf = post[1]
            cf = dictionary[term]
            ld = lengths[post[0]]
            results[post[0]] += language_model(tf, cf, cl, ld, mu)

    results = sorted(list(results.items()), key=lambda t: -t[1])
    # print(results, results[n-1][0])
    res = []
    for i in results[:n]:
        res.append(i[0])
    return res


class PreProcessor:

    def __init__(self):
        self.tokenizer = RegexpTokenizer(r'\w+')
        self.stop_words = set(stopwords.words('english'))

    def preprocess(self, text):
        bag_of_words = defaultdict(int)
        tokens = self.tokenizer.tokenize(text)
        for token in tokens:
            token = token.lower()
            if token not in self.stop_words:
                bag_of_words[token] += 1
        return bag_of_words


class Picklable:

    def __init__(self, path, name):
        self.path = path
        self.name = name

    def save(self):
        pickle.dump(self, open(self.path + '/' + self.name + '.p', 'wb'))

    @staticmethod
    def load(path):
        return pickle.load(open(path, 'rb'))


class DirectIndex(Picklable):
    name = 'direct_index'

    def __init__(self, path='index'):
        super(DirectIndex, self).__init__(path, self.name)
        self.index = {}

    def create(self, documents, pre_processor: PreProcessor):
        for document_id in documents:
            document = documents[document_id]
            self.index[document_id] = \
                pre_processor.preprocess(document)

    @staticmethod
    def load(path="index"):
        return Picklable.load(path + "/" + DirectIndex.name + ".p")


class InvertedIndex(Picklable):
    name = 'inverted_index'

    def __init__(self, path='index'):
        super(InvertedIndex, self).__init__(path, self.name)
        self.index = defaultdict(list)
        self.dictionary = defaultdict(int)
        self.cl = 0
        self.lengths = defaultdict(int)

    def create(self, direct_index):
        for document_id in direct_index:
            document = direct_index[document_id]
            for term in document:
                tf = direct_index[document_id][term]
                self.index[term] += [(document_id, tf)]
                self.cl += tf
                self.dictionary[term] += tf
                self.lengths[document_id] += tf

    @staticmethod
    def load(path="index"):
        return Picklable.load(path + "/" + InvertedIndex.name + ".p")


class Scorer:

    def __init__(self):
        self.args = {}

    def score(self, tf, df, cf, cl, ld):
        pass

    def set_arg(self, name, value):
        self.args[name] = value


class TFIDF(Scorer):

    def score(self, tf, df, cf, cl, ld):
        nD = self.args['D']
        return tf * math.log(nD / df)


class BM25(Scorer):

    def __init__(self, b, k1):
        super(BM25, self).__init__()
        self.b = b
        self.k1 = k1

    def score(self, tf, df, cf, cl, ld):
        nD = self.args['D']
        eld = cl / nD
        norm = self.k1 * (1 - self.b + self.b * ld / eld)
        return tf / (tf + norm) * math.log(nD / df)


class DirichletSmoothingLM(Scorer):

    def __init__(self, mu=100):
        super(DirichletSmoothingLM, self).__init__()
        self.mu = mu

    def score(self, tf, df, cf, cl, ld):
        return (tf + self.mu * cf / cl) / (self.mu + ld)


class Search:

    def __init__(self,
                 inverted_index: InvertedIndex,
                 documents,
                 pre_processor: PreProcessor,
                 scorer=DirichletSmoothingLM()):
        self.pre_processor = pre_processor
        self.inverted_index = inverted_index
        self.documents = documents
        self.scorer = scorer
        self.scorer.set_arg('D', len(documents))

    def set_indices(self, inverted_index: InvertedIndex, documents):
        self.inverted_index = inverted_index
        self.documents = documents
        self.scorer.set_arg('D', len(documents))

    def search(self, query, n=1, retrievable_paragraphs=None):
        query_bag_of_words = self.pre_processor.preprocess(query)
        query_bag_of_words = \
            [item for item in query_bag_of_words.items()
             if item[0] in self.inverted_index.dictionary]

        results = defaultdict(float)
        if retrievable_paragraphs is not None:
            results[min(retrievable_paragraphs)] = 0

        for term in query_bag_of_words:
            posting_list = self.inverted_index.index[term[0]]
            df = len(posting_list)
            for post in posting_list:
                if retrievable_paragraphs is None or post[0] in retrievable_paragraphs:
                    tf = post[1]
                    cf = self.inverted_index.dictionary[term]
                    ld = self.inverted_index.lengths[post[0]]
                    cl = self.inverted_index.cl
                    results[post[0]] += self.scorer.score(tf, df, cf, cl, ld)

        results = sorted(list(results.items()), key=lambda t: -t[1])

        res = []
        for i in results[:n]:
            res.append(i[0])
        return res
