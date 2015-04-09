'''
Created on 6 Apr 2015

@author: af
'''
from __future__ import division
import glob
import os
import numpy as np
from IPython.core.debugger import Tracer
import codecs
import nltk
from collections import defaultdict
from nltk.stem.snowball import EnglishStemmer
import pickle
import logging
import matplotlib.pyplot as plt
from datetime import datetime
import time
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
logging.info('script started at %s' %( str(datetime.now())))
script_start_time = time.time()
def getfilenames(base_dir):
    filenames = glob.glob(base_dir + '/*.txt')
    base_filenames = [os.path.basename(a_path) for a_path in filenames]
    doc_ids = range(len(base_filenames))
    docdict = dict(zip(doc_ids, base_filenames))
    return docdict

def getContent(filename, encoding='latin1'):
    with codecs.open(filename, mode='r', encoding=encoding) as inf:
        text = inf.read()
    return text

class Normaliser(object):
    def __init__(self, stemmer=None, stop_words=None, dictionary=None, lower_case=True):
        self.stemmer = stemmer
        self.stop_words = stop_words
        print("The number of stop words is %d" %(len(self.stop_words)))
        print("Stemmer is " + str(self.stemmer))
        self.dictionary = dictionary
        self.lower_case = lower_case
        
    def normalise(self, token):
        if self.lower_case:
            token = token.lower()
        if token in self.stop_words:
            return None
        if self.stemmer:
            token = self.stemmer.stem(token)
        if self.dictionary:
            if token not in self.dictionary:
                return None
        return token          

class Indexer(object):
    def __init__(self, tokeniser, normaliser=None):
        self.tokeniser = tokeniser
        self.normaliser = normaliser
        self.inverted_index = defaultdict(PostingList)
        #total number of documents
        self.N = 0
        self.document_lengths = defaultdict(float)
        
    def index(self, docID, text):
        tokens = self.tokeniser.tokenize(text)
        token_position = 0
        posting_lists = defaultdict(list)
        term_documents = defaultdict(TermDocument)
        for token in tokens:
            if self.normaliser:
                token = self.normaliser.normalise(token)
            if not token:
                continue
            term_document = term_documents[token]
            term_document.tf += 1
            term_document.positions.append(token_position)
            token_position += 1
        #update the main index
        for term, term_document in term_documents.iteritems():
            tf = term_document.tf
            self.document_lengths[docID] += np.square(tf)
            self.inverted_index[term].posts.append([docID, term_document])
            self.inverted_index[term].df += 1
        self.N += 1
        self.document_lengths[docID] = np.sqrt(self.document_lengths[docID])
    
    def dump(self, filename):
        logging.info("dumping index to %s" %(filename))
        with open(filename, 'wb') as outf:
            pickle.dump((self.inverted_index, self.document_lengths, self.N), outf)
    def load(self, filename):
        logging.info("loading index from %s" %(filename))
        with open(filename, 'rb') as inf:
            self.inverted_index, self.document_lengths, self.N = pickle.load(inf)
    def search(self, query):
        results = defaultdict(float)
        for term in query:
            posting_list = self.inverted_index[term]
            df = posting_list.df
            idf = np.log(self.N / (df+1))
            posts = posting_list.posts
            for post in posts:
                docID = post[0]
                term_document = post[1]
                tf = term_document.tf
                tfidf = tf * idf
                results[docID] += tfidf
        for docID in results:
            results[docID] = results[docID] / self.document_lengths[docID]
        ranked_results = sorted(results.items(), key=lambda x:x[1],reverse=True)
        return ranked_results
        
            

class QueryProcessor(object):
    def __init__(self, tokeniser, normaliser=None):
        self.tokeniser = tokeniser
        self.normaliser = normaliser
    
    def process(self, query):
        tokens = self.tokeniser.tokenize(query)
        query_terms = []
        for token in tokens:
            if normaliser:
                token = self.normaliser.normalise(token)
            if not token:
                continue
            query_terms.append(token)
        query_terms = self._expandQuery(query_terms)
        return query_terms
    
    def _expandQuery(self, query_terms):
        return query_terms
        
class PostingList(object):
    def __init__(self):
        # each post is like [docid, term_document] and a term_document is tf + positions
        self.posts = []
        #the number of documents a term occurred in.
        self.df = 0

class TermDocument(object):
    def __init__(self):
        #the number of times a term occurs in a document
        self.tf = 0
        self.positions = []
class NLTKWordTokenizer(object):
    def __init__(self):
        pass
    def tokenize(self, text):
        return nltk.word_tokenize(text)

def load_queries(query_file):
    logging.info("loading queries...")
    queries = {}
    with codecs.open(query_file, mode='r', encoding=encoding) as inf:
        for line in inf:
            fields = line.split('\t')
            queryID = int(fields[0])
            query = fields[1]
            queries[queryID] = query
    return queries

def dump_results(results_file, queries, query_results, docs):
    with open(results_file, 'wb') as outf:
        pickle.dump((queries, query_results, docs), outf)
    
def load_results(results_file):
    with open(results_file, 'rb') as inf:
        queries, query_results, docs = pickle.load(inf)
    return queries, query_results, docs

def load_relevant_docs(qrels_file, docname_id):
    query_relevant = defaultdict(list)
    with codecs.open(qrels_file, 'r', encoding=encoding) as inf:
        not_found = 0
        for line in inf:
            fields = line.split(' ')
            queryID = int(fields[0])
            docName = fields[2] + '.txt'
            if docName not in docname_id:
                not_found += 1
                logging.debug("%d not in docname_id: %s query_id: %d  relevance %s" %(not_found, docName, queryID, fields[3].strip()))
                continue
            docID = docname_id[docName ]
            docRelevance = int(fields[3].strip())
            if docRelevance > 0:
                query_relevant[queryID].append(docID)
    return query_relevant

def test_querying():
    query = "March of the Pinguins"
    query_terms = query_processor.process(query)
    results = indexer.search(query_terms)
    for result in results:
        docID, score = result
        docName = docs[docID]
        print("DocID: %d DocName: %s Score: %0.2f" %(docID, docName, score))

def print_a_file(docID):
    docName = docs[docID]
    filename = os.path.join(base_dir, docName)
    print(getContent(filename))

def get_docname_id(docs):
    docname_id = {}
    for id, docName in docs.iteritems():
        docname_id[docName] = id
    return docname_id   
def plot_numbers(x1, y1, plot_name):
    plot_file =   './' + plot_name + '.pdf'
    #x1 = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    #xs_labels = ['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9', '1.0']
    #ax = plt.gca()
    #y1 = [25.0, 28.3, 30.7, 31.2, 33.1, 36.0, 37.6, 38.8, 38.9, 39.7]

    #y_labels = ['GEOTEXT', 'Twitter-US', 'Twitter-WORLD']
    p1 = plt.plot(x1, y1, 'k.')
    #print np.mean(y1)

    #plt.axis().xaxis.set_ticks(xs)
    #plt.legend(('Mean Precision ' + suffix + ' = ' + str(np.mean(y1))), 'upper right', shadow=True)
    #plt.text(5, 0, 'Mean Precision ' + suffix + ' = ' + str(np.mean(y1)))
    plt.xlabel('Query Number')
    plt.ylabel('Precision@10')
    #ax.set_xticklabels(xs_labels)
    plt.title('Mean Precision ' + suffix + ' = ' + str(np.mean(y1)))
    #axis([0,2,-1,1])
    #plt.show(block=True)
    print "saving the plot in " + plot_file
    plt.savefig(plot_file, format='pdf')


def evaluate_precision_at_k(query_results, query_relevant, query=None, k=10):
    query_precision_at_k = {}
    for q, relevant in query_relevant.iteritems():
        if query!=None:
            if q!=query:
                continue
        if k > len(query_results[q]):
            logging.warn("k %d is larger than result set size %d! k set to %d" %(k, len(query_results[q]), len(query_results[q])))
            internal_k = len(query_results[q])
        else:
            internal_k = k
        retrieved_k_docs = query_results[q][0:internal_k]
        retrieved_k_docIDs = [docID for docID, score in retrieved_k_docs]
        tp=[docID for docID in retrieved_k_docIDs if docID in relevant]
        p = float(len(tp)) / internal_k 
        query_precision_at_k[q] = p
    if len(query_precision_at_k) == 1 and query!=None:
        return query_precision_at_k[query]
    logging.info("Average Precision in %d is %0.2f" %(k, np.mean(query_precision_at_k.values())))
    return query_precision_at_k
def evaluate_recall_at_k(query_results, query_relevant, k=10):
    query_recall_at_k = {}
    for q, relevant in query_relevant.iteritems():
        if k > len(query_results[q]):
            logging.warn("k %d is larger than result set size %d! k set to %d" %(k, len(query_results[q]), len(query_results[q])))
            internal_k = len(query_results[q])
        else:
            internal_k = k
        retrieved_k_docs = query_results[q][0:internal_k]
        retrieved_k_docIDs = [docID for docID, score in retrieved_k_docs]
        tp = [docID for docID in retrieved_k_docIDs if docID in relevant]
        recall = float(len(tp)) / len(relevant)
        query_recall_at_k[q] = recall
    
    logging.info("Average recall in %d is %0.2f" %(k, np.mean(query_recall_at_k.values())))
    return query_recall_at_k

def evaluate_map(query_results, query_relevant, depth_k=1000):
    query_map = {}
    for q, ret in query_results.iteritems():
        ret_docIDs = [docID for docID, score in ret]
        rel_docIDs = query_relevant[q]
        docRank = 1
        #m is the number of relevant documents found till now
        m = 0.0
        map_value = 0.0
        for docID in ret_docIDs:
            #document is relevant
            if docID in rel_docIDs:
                m += 1 
                precision_k = evaluate_precision_at_k(query_results, query_relevant, query=q, k=docRank)
                map_value += precision_k / m
            
            docRank += 1
            if docRank > len(ret_docIDs) or docRank > depth_k:
                break
            
        query_map[q] = map_value
    logging.info("Mean Average Precision is %0.2f" %( np.mean(query_map.values())))
    return query_map



#global parameters
suffix = 'nostem'
logging.info("creating the normaliser and the tokeniser...")
encoding = 'latin1'
stemmer = EnglishStemmer()
stemmer = None
normaliser = Normaliser(stemmer=stemmer, stop_words=nltk.corpus.stopwords.words('english'), dictionary=None, lower_case=True)  
tokeniser = NLTKWordTokenizer()
indexer = Indexer(tokeniser, normaliser)
query_processor = QueryProcessor(tokeniser, normaliser)
base_dir='./blogs'
index_file = os.path.join('./', 'index-' + suffix +'.pkl')
results_file = os.path.join('./', 'results-' + suffix + '.pkl')
qrels_file = os.path.join('./', 'qrels.february')
query_file = './06.topics.851-900.plain.final.txt'
logging.info("getting file names...")
docs = getfilenames(base_dir=base_dir)
docs_length = len(docs)



def main():
    queries = load_queries(query_file)
    RELOAD = False
    if not RELOAD:
        logging.info("Indexing %d docs in %s" %(docs_length, base_dir))
        docs_processed = 1
        tenpercent = int(docs_length / 10);
        for docID, filename in docs.iteritems():
            if docs_processed % tenpercent == 0:
                logging.info("processing " + str(10 * docs_processed / tenpercent) + "%")
            docs_processed += 1
            filename = os.path.join(base_dir, filename)
            text = getContent(filename, encoding=encoding)
            indexer.index(docID, text)
        indexer.dump(index_file)
    else:
        indexer.load(index_file)

    
    query_results = {}
    for queryID, query in queries.iteritems():
        query_terms = query_processor.process(query)
        results = indexer.search(query_terms)
        query_results[queryID] = results
    
    dump_results(results_file, queries, query_results, docs)

main()
queries, query_results, docs = load_results(results_file)
docname_id = get_docname_id(docs)
query_relevant = load_relevant_docs(qrels_file, docname_id)
precision_at_10 = evaluate_precision_at_k(query_results=query_results, query_relevant=query_relevant, k=10, query=None)
precision_at_100 = evaluate_precision_at_k(query_results=query_results, query_relevant=query_relevant, k=100, query=None)
plot_numbers(x1=precision_at_10.keys(), y1=precision_at_10.values(), plot_name='Precision@10-'+ suffix)
plot_numbers(x1=precision_at_100.keys(), y1=precision_at_100.values(), plot_name='Precision@100-'+ suffix)
query_map = evaluate_map(query_results, query_relevant)
query_recall_100 = evaluate_recall_at_k(query_results, query_relevant, k=100)
query_recall_10 = evaluate_recall_at_k(query_results, query_relevant, k=10)
#Tracer()()
script_end_time = time.time()
script_execution_hour = (script_end_time - script_start_time) / 60.0
logging.info("the script execution  is %s minutes" %(str(script_execution_hour)))