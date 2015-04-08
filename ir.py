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
logging.basicConfig(format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
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
    def __init__(self, stemmer=None, stop_words=None, dictionary=None):
        self.stemmer = stemmer
        self.stop_words = stop_words
        self.dictionary = dictionary
    def normalise(self, token):
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
        logging.info("logging index to %s" %(filename))
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
            idf = np.log(self.N / df)
            posts = posting_list.posts
            for post in posts:
                docID = post[0]
                term_document = post[1]
                tf = term_document.tf
                tfidf = tf * idf
                results[docID] += tfidf
        for docID in results:
            results[docID] = results[docID] / self.document_lengths[docID]
        ranked_results = sorted(results.items(), key=lambda x:x[1])
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
logging.info("creating the normaliser and the tokeniser...")
normaliser = Normaliser(EnglishStemmer(), nltk.corpus.stopwords.words('english'), dictionary=None)  
tokeniser = NLTKWordTokenizer()
indexer = Indexer(tokeniser, normaliser)
query_processor = QueryProcessor(tokeniser, normaliser)
base_dir='./blogs'
index_file = os.path.join('./', 'index.fast')
logging.info("getting file names...")
docs = getfilenames(base_dir=base_dir)
docs_length = len(docs)
logging.info("Indexing %d docs in %s" %(docs_length, base_dir))
docs_processed = 1
tenpercent = int(docs_length / 10);
for docID, filename in docs.iteritems():
    if docs_processed % tenpercent == 0:
        logging.info("\rprocessing " + str(10 * docs_processed / tenpercent) + "%")
    docs_processed += 1
    filename = os.path.join(base_dir, filename)
    text = getContent(filename, encoding='latin1')
    indexer.index(docID, text)

indexer.dump(index_file)
indexer.load(index_file)
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

Tracer()()