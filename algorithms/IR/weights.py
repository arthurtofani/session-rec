from collections import Counter, defaultdict
import numpy as np


def freq_weighted_items(sessions, index, item_df, current_session_items):
    x = [index[s] for s in sessions]
    x = sum(x, [])
    x = [i for i in x if i not in current_session_items]
    rr = dict(Counter(x).most_common())
    #rrr = [(k, rr[k] * np.log(len(item_df) / item_df[k])) for k in rr.keys()]
    return rr #dict(rrr)


def doc_weighted_items(sessions, rank, index):
    d = defaultdict(lambda: 0)
    for i, session in enumerate(sessions):
        for item in index[session]:
            d[item] += rank[i]
    return dict(d)


def tf_idf(terms, dict_tf, dict_df, tf_type, idf_type='default'):
    terms_tf = dict([(term, dict_tf[term]) for term in terms])
    terms_tf = TF_TYPES[tf_type](terms_tf)
    terms_idf = IDF_TYPES[idf_type](dict_df, terms)
    return dict([(k, terms_tf[k] * terms_idf[k]) for k in terms])


def tf_binary(terms_count):
    return dict([(term, 1) for term in terms_count])


def tf_raw(terms_count):
    return terms_count


def tf_log_norm(terms_count):
    return dict([(t, np.log(1 + v)) for t, v in terms_count.items()])


def default_idf(dict_df, terms):
    idf = lambda term: np.log(len(dict_df) / 1 + dict_df[term])
    return dict([(term, idf(term)) for term in terms])


def binary_idf(dict_df, terms):
    idf = lambda term: 1 if dict_df[term] > 0 else 0
    return dict([(term, idf(term)) for term in terms])


def binary(terms, *kwargs):
    # binary weightning is the same as tf=1, idf=1
    return dict([(k, 1) for k in terms])

TF_TYPES = {
    'binary': tf_binary,
    'raw': tf_raw,
    'log_norm': tf_log_norm
}

IDF_TYPES = {
    'binary': binary_idf,
    'default': default_idf,
}
