from _operator import itemgetter
from math import sqrt
import random
import time
import numpy as np
import pandas as pd
import os
import psutil
from tqdm import tqdm
from collections import defaultdict, Counter
from . import similarity as sim
from . import weights
import gc

class Sknn:
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

    def __init__( self, k, sample_size=1000, sampling='recent',  similarity = 'jaccard', remind=False, pop_boost=0, extend=False, normalize=True, session_key = 'SessionId', item_key= 'ItemId', time_key= 'Time' ):

        self.remind = remind
        self.k = k
        self.sample_size = sample_size
        self.similarity = similarity
        self.session_key = session_key
        self.item_key = item_key
        self.time_key = time_key
        self.normalize = normalize
        #updated while recommending
        self.term_index = dict()
        self.doc_index = dict()

        self.current_session_items = defaultdict(lambda: [])
        self.current_session_id = -1

    def fit_incremental(self, train, items=None):
        self.fit(train, items)

    def fit(self, train, items=None):
        docs_terms = train.groupby(self.session_key)[self.item_key].apply(set).to_dict()
        self.doc_index.update(docs_terms)
        terms_docs = train.groupby(self.item_key)[self.session_key].apply(set).to_dict()
        for term_id in terms_docs.keys():
            try:
                self.term_index[term_id].update(terms_docs[term_id])
            except KeyError:
                self.term_index[term_id] = set(terms_docs[term_id])



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

        if self.current_session_id != session_id:
            self.current_session_id = session_id
            self.current_session_items = []
        self.current_session_items.append(input_item_id)

        q_items = set(self.current_session_items)
        cand_sessions = list(self.select_candidates(q_items, self.term_index))
        if len(cand_sessions) > 0:

            # convert query and candidates to tf-idf bow representation
            query_bow = weights.binary(q_items)
            cand_bow = []
            for session in cand_sessions:
                session_items = self.doc_index[session]
                cand_bow.append(weights.binary(session_items))
            ranks = [sim.cos_similarity(query_bow, c) for c in cand_bow]

            weighted_items = weights.doc_weighted_items(cand_sessions,
                                                        ranks,
                                                        self.doc_index)
        else:
            weighted_items = {}
        scores = defaultdict(lambda: 0)
        scores.update(weighted_items)
        #import code; code.interact(local=dict(globals(), **locals()))
        return sim.ranked_results(predict_for_item_ids,
                                  scores,
                                  self.normalize)

    def select_candidates(self, terms, index):
        if len(terms) == 0:
            return set()
        return set().union(*[index.get(t, set()) for t in terms])[:self.k]

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
