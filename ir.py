'''
Created on 6 Apr 2015

@author: af
'''
from __future__ import division
import glob
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from IPython.core.debugger import Tracer
import codecs
import nltk
from collections import defaultdict, OrderedDict
from nltk.stem.snowball import EnglishStemmer
import pickle
import logging
import matplotlib.pyplot as plt

from datetime import datetime
import time
import matplotlib as mpl
mpl.use('Agg')

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
        self.dl = defaultdict(int)
        self.dic = defaultdict(int)
    def index(self, docID, text):
        tokens = self.tokeniser.tokenize(text)
        token_position = 0
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
            self.dl[docID] += 1
            self.dic[token] += 1
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
            pickle.dump((self.inverted_index, self.document_lengths, self.N, self.document_lengths, self.dl, self.dic), outf)
    def load(self, filename):
        logging.info("loading index from %s" %(filename))
        with open(filename, 'rb') as inf:
            self.inverted_index, self.document_lengths, self.N, self.document_lengths, self.dl, self.dic = pickle.load(inf)
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
    def search_logtf(self, query):
        results = defaultdict(float)
        for term in query:
            posting_list = self.inverted_index[term]
            df = posting_list.df
            idf = np.log(self.N / (df+1))
            posts = posting_list.posts
            for post in posts:
                docID = post[0]
                term_document = post[1]
                logtf = np.log(term_document.tf + 1)
                tfidf = logtf * idf
                results[docID] += tfidf
        for docID in results:
            results[docID] = results[docID] / self.document_lengths[docID]
        ranked_results = sorted(results.items(), key=lambda x:x[1],reverse=True)
        return ranked_results
    def search_bm25(self, query):
        results = defaultdict(float)
        k1=1.5
        b=0.75
        avgdl = np.mean(self.dl.values())
        for term in query:
            posting_list = self.inverted_index[term]
            df = posting_list.df
            idf = np.log((self.N - df + 0.5) / (df + 0.5))
            posts = posting_list.posts
            for post in posts:
                docID = post[0]
                term_document = post[1]
                tf = term_document.tf
                bm25 = idf * (tf * (k1 + 1)) / (tf + k1 * ( 1 - b + b * self.dl[docID] / avgdl))
                results[docID] += bm25

        ranked_results = sorted(results.items(), key=lambda x:x[1],reverse=True)
        return ranked_results
    def search_positional(self, query):
        if len(query) < 2:
            return None
        results = defaultdict(float)
        posting_lists = []
        for term in query:
            posting_list = self.inverted_index[term]
            posting_lists.append(posting_list)
        sorted_df_index = np.argsort([pl.df for pl in posting_lists]).tolist()
        for i in range(len(sorted_df_index)-1):
            first_plist = posting_lists[sorted_df_index[i]]
            next_plist = posting_lists[sorted_df_index[i+1]]
            for p1 in first_plist.posts:
                for p2 in next_plist.posts:
                    #if docIDs are equal
                    docID1 = p1[0]
                    docID2  = p2[0]
                    if docID1 == docID2:
                        poses1 = p1[1].positions
                        poses2 = p2[1].positions
                        for pos1 in poses1:
                            for pos2 in poses2:
                                if (pos1 - pos2) == (sorted_df_index[i] - sorted_df_index[i+1]):
                                    results[docID1] += 1.0 / (len(query)-1)
                                    break
                                elif (pos1 - pos2) < (sorted_df_index[i] - sorted_df_index[i+1]):
                                    break
                                elif (pos1 - pos2) > (sorted_df_index[i] - sorted_df_index[i+1]):
                                    continue
        ranked_results = sorted(results.items(), key=lambda x:x[1],reverse=True)
        if len(ranked_results) == 0:
            return None    
        return ranked_results
        
    def search_language_model(self,query, _lambda=0.5):       
        results = {}
        corpus_size = np.sum(self.dl.values())
        for term in query:
            p_t_collection = self.dic[term] / corpus_size
            posting_list = self.inverted_index[term]
            posts = posting_list.posts
            for post in posts:
                docID = post[0]
                term_document = post[1]
                tf = term_document.tf
                p_t_d = tf / self.dl[docID]
                lm = _lambda * p_t_d + (1 - _lambda) * p_t_collection
                if lm==0:
                    logging.warn("LM score is 0.0 , this is a severe error.")
                if docID in results:
                    results[docID] *= lm
                else:
                    results[docID] = lm

        ranked_results = sorted(results.items(), key=lambda x:x[1],reverse=True)
        return ranked_results

    def search_language_model_laplace_smoothing(self,query):       
        results = defaultdict(float)
        qtf = 1
        avgdl = np.sum(self.dl.values())
        V = len(self.dic)
        corpus_size = np.sum(self.dl.values())
        for term in query:
            p_t_collection = self.dic[term] / corpus_size
            posting_list = self.inverted_index[term]
            posts = posting_list.posts
            for post in posts:
                docID = post[0]
                term_document = post[1]
                tf = term_document.tf
                lm = qtf * np.log( (tf + 1) / (self.dl[docID] + V))
                if lm==0:
                    logging.warn("LM score is 0.0 , this is a severe error.")
                results[docID] += lm

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
def plot_numbers(x1, y1, plot_name, x_label, y_label, p_title, marker='k.'):
    plot_file =   './' + plot_name + '.pdf'
    p1 = plt.plot(x1, y1, marker)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(p_title)
    print "saving the plot in " + plot_file
    plt.savefig(plot_file, format='pdf')
    plt.close()


def evaluate_precision_at_k(query_results, query_relevant, query=None, k=10):
    query_precision_at_k = {}
    for q, relevant in query_relevant.iteritems():
        if query!=None:
            if q!=query:
                continue
        if k > len(query_results[q]):
            logging.debug("k %d is larger than result set size %d! k set to %d" %(k, len(query_results[q]), len(query_results[q])))
            internal_k = len(query_results[q])
        else:
            internal_k = k
        if internal_k == 0:
            query_precision_at_k[q] = 0.0
            logging.debug("Query %d has no retrieved docs" %(q))
        else:
            retrieved_k_docs = query_results[q][0:internal_k]
            retrieved_k_docIDs = [docID for docID, score in retrieved_k_docs]
            tp=[docID for docID in retrieved_k_docIDs if docID in relevant]             
            p = float(len(tp)) / internal_k 
            query_precision_at_k[q] = p
    if len(query_precision_at_k) == 1 and query!=None:
        return query_precision_at_k[query]
    logging.info("Average Precision in %d is %0.2f" %(k, np.mean(query_precision_at_k.values())))
    return query_precision_at_k
def evaluate_recall_at_k(query_results, query_relevant, query=None, k=10):
    query_recall_at_k = {}
    for q, relevant in query_relevant.iteritems():
        if query!=None:
            if q!=query:
                continue
        if k > len(query_results[q]):
            logging.debug("k %d is larger than result set size %d! k set to %d" %(k, len(query_results[q]), len(query_results[q])))
            internal_k = len(query_results[q])
        else:
            internal_k = k
        if internal_k == 0:
            query_recall_at_k[q] = 0
            logging.debug("Query %d has no retrieved docs" %(q))
        else:
            retrieved_k_docs = query_results[q][0:internal_k]
            retrieved_k_docIDs = [docID for docID, score in retrieved_k_docs]
            tp = [docID for docID in retrieved_k_docIDs if docID in relevant]
            recall = float(len(tp)) / len(relevant)
            query_recall_at_k[q] = recall
    if len(query_recall_at_k) == 1 and query!=None:
        return query_recall_at_k[query]
    logging.info("Average recall in %d is %0.2f" %(k, np.mean(query_recall_at_k.values())))
    return query_recall_at_k

def evaluate_map(query_results, query_relevant, depth_k=1000):
    query_map = {}
    for q, ret in query_results.iteritems():
        ret_docIDs = [docID for docID, score in ret]
        rel_docIDs = query_relevant[q]
        docRank = 1
        #m is the number of relevant documents found till now
        m = 0
        map_value = 0.0
        for docID in ret_docIDs:
            #document is relevant
            if docID in rel_docIDs:
                m += 1 
                precision_k = evaluate_precision_at_k(query_results, query_relevant, query=q, k=docRank)
                map_value += precision_k
            
            docRank += 1
            if docRank > len(ret_docIDs) or docRank > depth_k:
                break
        if m!= 0:
            map_value = map_value / m    
        query_map[q] = map_value
    logging.info("Mean Average Precision is %0.2f" %( np.mean(query_map.values())))
    return query_map
def evaluate_pr_curve(query_results, query_relevant, dump_file, reload=False):
    if reload and os.path.exists(dump_file):
        with open(dump_file, 'rb') as inf:
            rp_fianl = pickle.load(inf)
            return rp_fianl
    rp_final = OrderedDict()
    r_p_allq = defaultdict(list)
    for q, ret in query_results.iteritems():
        r_p = defaultdict(list)
        for i in range(len(ret)):
            p = evaluate_precision_at_k(query_results, query_relevant, query=q, k=(i+1))
            r = evaluate_recall_at_k(query_results, query_relevant, query=q, k=(i+1))
            aggregated_r = (10 * r) // 1
            r_p[aggregated_r].append(p)
        for r, ps in r_p.iteritems():
            maxp = np.max(ps)
            r_p_allq[r].append(maxp)
    for r in sorted(r_p_allq):
        rp_final[r] = np.mean(r_p_allq[r])
    with open(dump_file, 'wb') as inf:
        pickle.dump(rp_final, inf)
    return rp_final
def aggregate_numbers(dic):
    newdic = defaultdict(list)
    aggregateddic = OrderedDict()
    for k,v in dic.iteritems():
        newk = int(100 * k) / 100.0
        newdic[newk].append(v)
    for k in sorted(newdic):
        v = newdic[k]
        maxv = np.max(v)
        aggregateddic[k] = maxv
    return aggregateddic
def sort_by_value(data):
    #sort by value
    kv_list = sorted(data.items(), key=lambda x:x[1])
    sorted_dic = OrderedDict()
    for kv in kv_list:
        k, v = kv
        sorted_dic[k] = v
    return sorted_dic
#global parameters

def rp_curves(rps):
    stem_rp = rps['stem']
    stem_logtf_rp = rps['stem-logtf']
    #stem_positional_rp = rps['stem-positional']
    stem_positional_merged_rp = rps['stem-positional-merged']
    stem_bm25_rp = rps['stem-bm25']
    labels = ('TF-IDF', 'LOGTF-IDF', 'TFIDF+POSITIONAL', 'BM25')
    
    p1 = plt.plot([a/10 for a in stem_rp.keys()], stem_rp.values(), 'r-')
    p2 = plt.plot([a/10 for a in stem_logtf_rp.keys()], stem_logtf_rp.values(), 'y-')
    #p3 = plt.plot([a/10 for a in stem_positional_rp.keys()], stem_positional_rp.values(), 'b-')
    p4 = plt.plot([a/10 for a in stem_positional_merged_rp.keys()], stem_positional_merged_rp.values(), 'm-')
    p5 = plt.plot([a/10 for a in stem_bm25_rp.keys()], stem_bm25_rp.values(), 'k-')
    #plt.axis().xaxis.set_ticks(xs)
    plt.legend(labels, 'upper right', shadow=True)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    #ax.set_xticklabels(xs_labels)
    plt.title("RP Curve: Effect of Scoring/Weighting Schemes")
    plt.show(block=True)
    plt.savefig('rp_scoring.pdf', format='pdf')

def bar_chart():    
    N = 5
    tfidf = (0.52, 0.48, 0.51, 0.84, 0.07)
    logtfidf = (0.43, 0.45, 0.45, 0.83, 0.06)
    #positional = (0.24, 0.23, 0.41, 0.44, 0.05)
    tfidf_positional = (0.64, 0.53, 0.59, 0.87, 0.09)
    bm25 = (0.66, 0.54, 0.61, 0.86, 0.10)
    
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.1       # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, tfidf, width, color='r')
    rects2 = ax.bar(ind+width, logtfidf, width, color='y')
    #rects3 = ax.bar(ind+2*width, positional, width, color='b')
    rects3 = ax.bar(ind+2*width, tfidf_positional, width, color='m')
    rects4 = ax.bar(ind+3*width, bm25, width, color='k')
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('Precision/Recall')
    ax.set_title('Effect of Scoring/Weighting schemes on Evaluation Measures')
    ax.set_xticks(ind+width)
    ax.set_xticklabels( ('P@10', 'P@100', 'MAP', 'R@1000', 'R@10') )
    
    ax.legend( (rects1[0], rects2[0], rects3[0], rects4[0]), ('TF-IDF', 'LOGTF-IDF', 'TFIDF+POSITIONAL', 'BM25') , loc='best', fontsize='small')
    
    def autolabel(rects):
        # attach some text labels
        for rect in rects:
            height = rect.get_height()
            ax.text(rect.get_x()+rect.get_width()/2., 1.05*height, '%d'%int(height),
                    ha='center', va='bottom')
    

    plt.savefig('bar_weightings.pdf', format='pdf')
def main(methods=['bm25']):
    queries = load_queries(query_file)
    RELOAD = True
    if not RELOAD or not os.path.exists(index_file):
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

    logging.info('Search started.')
    query_results = {}
    for queryID, query_unprocessed in queries.iteritems():
        query_terms = query_processor.process(query_unprocessed)
        results_positional = None
        results_tfidf = None
        results_logtfidf = None
        results_bm25 = None
        results_lm = None
        results_lmls = None
        ['stem', 'stem-logtf', 'stem-positional-merged', 'stem-bm25']
        if 'stem' in methods:
            results_tfidf = indexer.search(query_terms)
        if 'stem-logtf' in methods:
            results_logtfidf = indexer.search_logtf(query_terms)
        if 'stem-positional' in methods:
            results_positional = indexer.search_positional(query_terms)
        if 'stem-positional-merged' in methods:
            results_positional = indexer.search_positional(query_terms)
            results_tfidf = indexer.search(query_terms)
        if 'stem-bm25' in methods:
            results_bm25 = indexer.search_bm25(query_terms)
        if 'stem-lm' in methods:
            results_lm = indexer.search_language_model(query_terms, _lambda=0.9)
        if 'stem-lmls' in methods:
            results_lmls = indexer.search_language_model_laplace_smoothing(query_terms)
        results_merged = defaultdict(float)
        if results_positional is not None:
            for docID, score in results_positional:
                results_merged[docID] += score
        if results_tfidf is not None:
            for docID, score in results_tfidf:
                results_merged[docID] += score
        if results_logtfidf is not None:
            for docID, score in results_logtfidf:
                results_merged[docID] += score
        if results_bm25 is not None:
            for docID, score in results_bm25:
                results_merged[docID] += score
        if results_lm is not None:
            for docID, score in results_lm:
                results_merged[docID] += score
        if results_lmls is not None:
            for docID, score in results_lmls:
                results_merged[docID] += score
        ranked_results = sorted(results_merged.items(), key=lambda x:x[1],reverse=True)
        query_results[queryID] = ranked_results
    logging.info('Search finished.')

    dump_results(results_file, queries, query_results, docs)


DIAGRAMS = False
rps = {}
for suffix in ['stem', 'stem-logtf', 'stem-positional-merged', 'stem-bm25']:#['stem', 'stem-logtf','stem-positional', 'stem-positional-merged', 'stem-bm25']
    logging.info("creating the normaliser and the tokeniser...")
    encoding = 'latin1'
    stemmer = EnglishStemmer()
    #stemmer = None
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
    '''
    if methods in main(methods) is set to multiple methods like ['stem', 'stem-bm25'] 
    then the scores of each method would be summed up in the final results.
    '''
    main(methods=[suffix])
    queries, query_results, docs = load_results(results_file)
    docname_id = get_docname_id(docs)
    query_relevant = load_relevant_docs(qrels_file, docname_id)
    precision_at_10 = evaluate_precision_at_k(query_results=query_results, query_relevant=query_relevant, k=10, query=None)
    ranked_results = sorted(precision_at_10.items(), key=lambda x:x[1],reverse=True)
    precision_at_100 = evaluate_precision_at_k(query_results=query_results, query_relevant=query_relevant, k=100, query=None)
    

    query_map = evaluate_map(query_results, query_relevant)
    query_recall_1000 = evaluate_recall_at_k(query_results, query_relevant, query=None, k=1000)
    query_recall_100 = evaluate_recall_at_k(query_results, query_relevant, query=None, k=100)
    query_recall_10 = evaluate_recall_at_k(query_results, query_relevant, query=None, k=10)
    rp_final = evaluate_pr_curve(query_results, query_relevant, dump_file='rp_final-'+ suffix + '.pkl', reload=True)
    rps[suffix] = rp_final
    if DIAGRAMS:
        plot_numbers(x1=precision_at_10.keys(), y1=precision_at_10.values(), plot_name='Precision@10-'+ suffix, x_label='Query Number', y_label='Precision@10',p_title='Precision@10 ' + suffix + ' Avg = ' + str(np.mean(precision_at_10.values())), marker='k.')
        plot_numbers(x1=precision_at_100.keys(), y1=precision_at_100.values(), plot_name='Precision@100-'+ suffix, x_label='Query Number', y_label='Precision@100',p_title='Precision@100 ' + suffix + ' Avg = ' + str(np.mean(precision_at_100.values())), marker='k.')
        plot_numbers(rp_final.keys(), rp_final.values(), plot_name='p-r-' + suffix, marker='r-', x_label='Recall', y_label='Precision',p_title='PR Curve ' + suffix )

if DIAGRAMS:
    rp_curves(rps)
    bar_chart()
script_end_time = time.time()
script_execution_hour = (script_end_time - script_start_time) / 60.0
logging.info("the script execution  is %s minutes" %(str(script_execution_hour)))