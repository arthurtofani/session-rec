import rankaggregation as ra  # pip install git+https://github.com/djcunningham0/rankaggregation.git
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from . import similarity as sim
from . import weights
#import time


class Bow():

    @classmethod
    def from_dict(cls, dic):
        b = cls(dic.keys())
        b.vec = dic
        return b

    def __init__(self, components):
        self.components = components
        self.vec = None

    def rem(self, elems):
        if self.vec:
            [self.vec.pop(k, None) for k in elems]
            self.components = [s for s in self.components if s not in elems]
        return self

    def top(self, n):
        return dict(Counter(self.vec).most_common(n))

    def norm(self):
        if len(self.vec) == 0:
            return self
        maxval = max(self.vec.values())
        return Bow.from_dict({k: self.vec[k] / maxval for k in self.vec})

    def multiply(self, bow):
        intersec = set(self.vec.keys()) & set(bow.vec.keys())
        return Bow.from_dict({k: self.vec[k] * bow.vec[k] for k in intersec})

    def sum(self, bow):
        intersec = set(self.vec.keys()) & set(bow.vec.keys())
        return Bow.from_dict({k: self.vec[k] + bow.vec[k] for k in intersec})


    def tf_idf(self, dict_tf, dict_df, tf_type, idf_type):
        vec = weights.tf_idf(self.components, dict_tf, dict_df,
                             tf_type, idf_type)
        return Bow.from_dict(vec)

    def sequential_alignment(self, session_items):
        els, intsc, _ = np.intersect1d(self.components, session_items, return_indices=True)
        s = np.zeros(len(self.components))
        arr = np.arange(len(self.components))
        for i in intsc:
            s += (arr - i)**2
        s = 1/(1 + s)
        return Bow.from_dict(dict(zip(self.components, s))).rem(els)


    def compare(self, target_vecs, sim_method):
        ranks = [sim_method(self.vec, c.vec) for c in target_vecs]
        return target_vecs, ranks

    def truncate_at_k(self, target_vecs, ranks, k):
        idxs = np.array(ranks).argsort()[:k]
        return np.array(target_vecs)[idxs], np.array(ranks)[idxs]





class IRModel:
    '''
    TfIdf( k, sample_size=500, session_key = 'SessionId', item_key= 'ItemId')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    extend : bool
        Add evaluated sessions to the maps
    normalize : bool
        Normalize the scores in the end
    session_key : string
        Header of the session ID column in the input file. (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file. (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file. (default: 'Time')
    '''

    def __init__(self, k, sample_size=300, remind=False, pop_boost=0,
                 extend=False, normalize=True, session_key='SessionId',
                 aggregate_method='borda',
                 item_ranking_method="sequential_alignment",
                 item_key='ItemId', tf_type='binary', idf_type='binary',
                 cut=5,
                 timestamp=None ):

        self.remind = remind
        self.k = k
        self.sample_size = sample_size
        self.cut = cut
        self.tf_type = tf_type
        self.aggregate_method = aggregate_method
        self.item_ranking_method = item_ranking_method
        self.idf_type = idf_type
        self.session_key = session_key
        self.item_key = item_key
        self.normalize = normalize
        self.term_index = dict()
        self.doc_index = dict()

        self.item_tf = defaultdict(lambda: {})   # key => [doc][term]
        self.item_df = defaultdict(lambda: 0)  # key => (term)
        self.curr_items = defaultdict(lambda: [])
        self.curr_session_id = -1

    def fit_incremental(self, train, items=None):
        self.fit(train, items)

    def fit(self, train, items=None, init=False):
        #import code; code.interact(local=dict(globals(), **locals()))
        docs_terms = train.groupby(self.session_key)[self.item_key].apply(list).to_dict()
        self.doc_index.update(docs_terms)
        terms_docs = train.groupby(self.item_key)[self.session_key].apply(list).to_dict()
        for term_id in terms_docs.keys():
            for doc_id, freq in Counter(terms_docs[term_id]).most_common():
                self.item_df[term_id] += 1
                self.item_tf[doc_id][term_id] = freq
            try:
                self.term_index[term_id].update(terms_docs[term_id])
            except KeyError:
                self.term_index[term_id] = set(terms_docs[term_id])

    def top_k(self, cand_bows, scores, k=None):
        return sorted(zip(cand_bows, scores), reverse=True, key=lambda x: x[1])[:k]

    def aggregate_rank(self, dicts, rank_method=None, cut=20):
        if rank_method is None:
            rank_method = ra.RankAggregator().borda
        rank_keys = lambda d: [x for x, _ in sorted(d.vec.items(), key=lambda x: x[1], reverse=True)][:cut]
        return dict(rank_method([rank_keys(n) for n in dicts]))

    def update_session(self, session_id, input_item_id):
        if self.curr_session_id != session_id:
            self.curr_session_id = session_id
            self.curr_items = []
        self.curr_items.append(input_item_id)

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, mode_type='view', timestamp=0):
        self.update_session(session_id, input_item_id)
        query = set(self.curr_items)
        #import code; code.interact(local=dict(globals(), **locals()))

        cand_sessions = self.select_candidates(query, self.term_index)
        query_bow, cand_bows = self.tf_idf(query, cand_sessions, self.item_df,
                                           self.tf_type, self.idf_type)
        if len(cand_sessions) > 0:
            _, scores = query_bow.compare(cand_bows, sim.cos_similarity)
            top_k_tuples = self.top_k(cand_sessions, scores, k=self.k)
            top_sessions = [x for x, _ in top_k_tuples]

            if len(top_sessions) > 0:
                if self.item_ranking_method == "sequential_alignment":
                    timeord_its = [Bow(self.doc_index[d]).sequential_alignment(self.curr_items) for d in top_sessions]
                    if self.aggregate_method == "borda":
                        bow_time = Bow.from_dict(self.aggregate_rank(timeord_its, cut=self.cut)).norm()
                    if self.aggregate_method == "sum":
                        vec_x = timeord_its[0].vec
                        for bow_y in range(1, len(timeord_its)):
                            vec_y = timeord_its[bow_y].vec
                            vec_x = {k: vec_x.get(k, 0) + vec_y.get(k, 0) for k in set(vec_x) | set(vec_y)}
                        bow_time = Bow.from_dict(vec_x)
                    if self.aggregate_method == "multiply":
                        vec_x = timeord_its[0].vec
                        for bow_y in range(1, len(timeord_its)):
                            vec_y = timeord_its[bow_y].vec
                            vec_x = {k: vec_x.get(k, 0) * vec_y.get(k, 0) for k in set(vec_x) | set(vec_y)}
                        bow_time = Bow.from_dict(vec_x)
                        #import code; code.interact(local=dict(globals(), **locals()))
                        #pass
                else:
                    pass
                    #bow_time = Bow.from_dict(self.aggregate_rank(timeord_its)).norm()
                items_scores = bow_time.vec
            else:
                items_scores = {}
        else:
            items_scores = {}

        scores = defaultdict(lambda: 0)
        scores.update(items_scores)

        predictions = np.zeros(len(predict_for_item_ids))
        mask = np.in1d(predict_for_item_ids, list(scores.keys()))

        items = predict_for_item_ids[mask]
        values = [scores[x] for x in items]
        predictions[mask] = values
        series = pd.Series(data=predictions, index=predict_for_item_ids)

        if self.normalize:
            series = series / series.max()

        return series

    def tf_idf(self, q_items, cand_sessions, dict_df, tf_type, idf_type):
        if len(cand_sessions) > 0:

            # convert query and candidates to tf-idf bow representation
            query_bow = Bow(q_items).tf_idf(Counter(q_items), dict_df,
                                            tf_type, idf_type)
            cand_bows = []
            for s in cand_sessions:
                session_items = self.doc_index[s]
                cand_bow = Bow(session_items).tf_idf(self.item_tf[s],
                                                     self.item_df,
                                                     tf_type, idf_type)
                cand_bows.append(cand_bow)
            return query_bow, cand_bows
        else:
            return None, []

    def select_candidates(self, terms, index):
        if len(terms) == 0:
            return set()
        return list(set().union(*[index.get(t, set()) for t in terms]))

    def clear(self):
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()
