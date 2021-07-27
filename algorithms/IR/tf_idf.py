import rankaggregation as ra  # pip install git+https://github.com/djcunningham0/rankaggregation.git
import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from . import similarity as sim
from . import weights


class Bow():

    @classmethod
    def from_dict(cls, dic):
        b = cls(dic.keys())
        b.vec = dic
        return b

    def __init__(self, components):
        self.components = components
        self.vec = None

    def norm(self):
        if len(self.vec) == 0:
            return self
        maxval = max(self.vec.values())
        return Bow.from_dict({k: self.vec[k] / maxval for k in self.vec})

    def multiply(self, bow):
        intersec = set(self.vec.keys()) & set(bow.vec.keys())
        return Bow.from_dict({k: self.vec[k] * bow.vec[k] for k in intersec})

    def tf_idf(self, dict_tf, dict_df, tf_type, idf_type):
        vec = weights.tf_idf(self.components, dict_tf, dict_df,
                             tf_type, idf_type)
        return Bow.from_dict(vec)

    def time_seq(self):
        sz = [x/len(self.components) for x in range(len(self.components))]
        return Bow.from_dict(dict(zip(self.components, sz)))


    def compare(self, target_vecs, sim_method):
        ranks = [sim_method(self.vec, c.vec) for c in target_vecs]
        return target_vecs, ranks

    def truncate_at_k(self, target_vecs, ranks, k):
        idxs = np.array(ranks).argsort()[:k]
        return np.array(target_vecs)[idxs], np.array(ranks)[idxs]





class TfIdf:
    '''
    TfIdf( k, sample_size=500, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, session_key = 'SessionId', item_key= 'ItemId')

    Parameters
    -----------
    k : int
        Number of neighboring session to calculate the item scores from. (Default value: 100)
    sample_size : int
        Defines the length of a subset of all training sessions to calculate the nearest neighbors from. (Default value: 500)
    sampling : string
        String to define the sampling method for sessions (recent, random). (default: recent)
    similarity : string
        String to define the method for the similarity calculation (jaccard, cosine, binary, tanimoto). (default: jaccard)
    remind : bool
        Should the last items of the current session be boosted to the top as reminders
    pop_boost : int
        Push popular items in the neighbor sessions by this factor. (default: 0 to leave out)
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

    def __init__( self, k, sample_size=1000, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, extend=False, normalize=True, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time', tf_type='raw', idf_type='idf' ):

        self.remind = remind
        self.k = k
        self.sample_size = sample_size
        self.tf_type = tf_type
        self.idf_type = idf_type
        self.similarity = similarity
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.normalize = normalize
        #updated while recommending
        self.term_index = dict()
        self.doc_index = dict()

        self.item_tf = defaultdict(lambda: {})   # key => [doc][term]
        self.item_df = defaultdict(lambda: 0)  # key => (term)
        self.curr_items = defaultdict(lambda: [])
        self.curr_session_id = -1


    def fit_incremental(self, train, items=None):
        self.fit(train, items)


    def fit(self, train, items=None):
        from collections import OrderedDict
        listset = lambda x: list(set(x))
        listset2 = lambda x: list(OrderedDict.fromkeys(x))
        docs_terms = train.groupby(self.session_key)[self.item_key].apply(listset2).to_dict()
        self.doc_index.update(docs_terms)
        #print(len(self.doc_ndex))
        terms_docs = train.groupby(self.item_key)[self.session_key].apply(listset).to_dict()
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


    def aggregate_rank(self, dicts, rank_method=None):

        if rank_method is None:
            rank_method = ra.RankAggregator().borda
        #import code; code.interact(local=dict(globals(), **locals()))
        rank_keys = lambda d: [x for x, _ in sorted(d.vec.items(), key=lambda x: x[1], reverse=True)]
        return dict(rank_method([rank_keys(n) for n in dicts]))

    def predict_next(self, session_id, input_item_id, predict_for_item_ids, skip=False, mode_type='view', timestamp=0):
        '''
        Gives predicton scores for a selected set of items on how likely they be the next item in the session.

        Parameters
        --------
        session_id : int or string
            The session IDs of the event.
        input_item_id : int or string
            The item ID of the event. Must be in the set of item IDs of the training set.
        predict_for_item_ids : 1D array
            IDs of items for which the network should give prediction scores. Every ID must be in the set of item IDs of the training set.

        Returns
        --------
        out : pandas.Series
            Prediction scores for selected items on how likely to be the next item of this session. Indexed by the item IDs.

        '''

        if self.curr_session_id != session_id:
            self.curr_session_id = session_id
            self.curr_items = []
        self.curr_items.append(input_item_id)

        query = set(self.curr_items)
        cand_sessions = self.select_candidates(query, self.term_index)

        query_bow, cand_bows = self.tf_idf(query, cand_sessions, self.item_df,
                                           self.tf_type, self.idf_type)

        _, scores = query_bow.compare(cand_bows, sim.cos_similarity)

        top_k_tuples = self.top_k(cand_sessions, scores, k=self.k)
        top_sessions = [x for x, _ in top_k_tuples]
        #top_ses_rank = [x for _, x in top_k_tuples]

        if len(top_sessions) > 0:
            #import code; code.interact(local=dict(globals(), **locals()))

            #cand_items = self.select_candidates(top_sessions, self.doc_index)

            # A. Items ranked by top session candidate scores
            doc_scores = [1-(x/len(top_sessions)) for x in range(len(top_sessions))]
            weighted_items = weights.doc_weighted_items(top_sessions,
                                                        doc_scores,
                                                        self.doc_index)
            bow_doc_weights = Bow.from_dict(weighted_items).norm()

            # B. Items ranked by frequency in top candidates
            freqs = weights.freq_weighted_items(top_sessions, self.doc_index,
                                                self.item_df, self.curr_items)
            bow_freq = Bow.from_dict(freqs).norm()

            # C. Items ranked by local (top sessions) items idf
            top_sess_its = [list(set(self.doc_index[d])) for d in top_sessions]
            top_docs_idf = Counter(np.hstack(top_sess_its))
            bow_idf = Bow.from_dict(top_docs_idf).norm()


            # D. Items ranked by position in session (best-scored at end position)
            timeord_its = [Bow(self.doc_index[d]).time_seq() for d in top_sessions]
            bow_time = Bow.from_dict(self.aggregate_rank(timeord_its)).norm()

            vecs = [bow_freq, bow_idf, bow_doc_weights, bow_doc_weights, bow_time, bow_time]
            items_scores = self.aggregate_rank(vecs)

        else:
            #weighted_items = {}
            items_scores = {}
            #ff = {}

        scores = defaultdict(lambda: 0)
        #scores.update(weighted_items)
        scores.update(items_scores)

        #import code; code.interact(local=dict(globals(), **locals()))

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

            #query_bow = Bow(q_items).time_seq()
            cand_bows = []
            for s in cand_sessions:
                session_items = self.doc_index[s]
                cand_bow = Bow(session_items).tf_idf(self.item_tf[s],
                                                     self.item_df,
                                                     tf_type, idf_type)

                #cand_bow1 = Bow(session_items).time_seq()
                cand_bows.append(cand_bow)
            return query_bow, cand_bows
        else:
            return None, []

    def select_candidates(self, terms, index):
        if len(terms) == 0:
            return set()
        return list(set().union(*[index.get(t, set()) for t in terms]))

    def score_items(self, neighbors):
        '''
        Compute a set of scores for all items given a set of neighbors.

        Parameters
        --------
        neighbors: set of session ids

        Returns
        --------
        out : list of tuple (item, score)
        '''
        # now we have the set of relevant items to make predictions
        scores = dict()
        # iterate over the sessions
        for session in neighbors:
            # get the items in this session
            items = self.items_for_session( session[0] )

            for item in items:
                old_score = scores.get( item )
                new_score = session[1]

                if old_score is None:
                    scores.update({item : new_score})
                else:
                    new_score = old_score + new_score
                    scores.update({item : new_score})

        return scores



    def clear(self):
        self.session = -1
        self.session_items = []
        self.relevant_sessions = set()

        self.session_item_map = dict()
        self.item_session_map = dict()
        self.session_time = dict()
