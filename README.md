# irproject1
A simple information retrieval project using inverted index and vector space models.

1)The source code is just one python file ir.py.

2)The code is written with Python 2.7.

3)The query_file and base_dir variables in the code should be set to the query file and the blogs directory respectively.

4)The query file should have this format:
851	March of the Penguins
one query at each line with no quotations.

5)Each implemented IR model has a suffix (name).
TFIDF:  'stem'
LogtfIDF: 'stem-logtf'
TFIDF-Positional: 'stem-positional'
BM25: 'stem-bm25'
language model: 'stem-lm'
language model with laplace smoothing: 'stem-lmls'

In order to linearly sum two scores from two models, it is possible to set methods in main(methods) to be a list of
suffixes that we intend to be used in the final results. For example main(methods=['stem', 'stem-positional']) sums up
the socres of two different search strategies and returns the final results.

6)Results would be written to results_file which should be set in the code. The result format
is a pickle file that contains 3 variables. It should be loaded using pickle like this:
    with open(results_file, 'rb') as inf:
        queries, query_results, docs = pickle.load(inf)
    return queries, query_results, docs
There exists a function load_results that does this for the user:
load_results(results_file)


For further information please contact me at  afshinrahimi@gmail.com


