import time
import numpy as np
from tqdm import tqdm


def evaluate_sessions(pr, metrics, test_data, train_data, day, items=None, cut_off=20, session_key='SessionId', item_key='ItemId', time_key='Time'):
    '''
    Evaluates the baselines wrt. recommendation accuracy measured by recall@N and MRR@N. Has no batch evaluation capabilities. Breaks up ties.

    Parameters
    --------
    pr : baseline predictor
        A trained instance of a baseline predictor.
    metrics : list
        A list of metric classes providing the proper methods
    test_data : pandas.DataFrame
        Test data. It contains the transactions of the test set.It has one column for session IDs, one for item IDs and one for the timestamp of the events (unix timestamps).
        It must have a header. Column names are arbitrary, but must correspond to the keys you use in this function.
    train_data : pandas.DataFrame
        Training data. Only required for selecting the set of item IDs of the training set.
    items : 1D list or None
        The list of item ID that you want to compare the score of the relevant item to. If None, all items of the training set are used. Default value is None.
    cut-off : int
        Cut-off value (i.e. the length of the recommendation list; N for recall@N and MRR@N). Defauld value is 20.
    session_key : string
        Header of the session ID column in the input file (default: 'SessionId')
    item_key : string
        Header of the item ID column in the input file (default: 'ItemId')
    time_key : string
        Header of the timestamp column in the input file (default: 'Time')

    Returns
    --------
    out :  list of tuples
        (metric_name, value)

    '''
    import pandas as pd

    actions = len(test_data)
    sessions = len(test_data[session_key].unique())
    count = 0
    print('START evaluation of ', actions, ' actions in ', sessions, ' sessions')

    sc = time.clock()
    st = time.time()

    time_sum = 0
    time_sum_clock = 0
    time_count = 0

    for m in metrics:
        m.reset()

    #test_data['day'] = (pd.to_datetime(test_data.Time, unit='s') - pd.Timestamp(0)).dt.days
    #test_data['day'] = test_data.groupby('SessionId')['day'].transform('min')
    #days = test_data.groupby('day').groups.keys()


    test_data = test_data[test_data.day == day].head(1000)
    test_data.sort_values([session_key, time_key], inplace=True)
    items_to_predict = train_data[item_key].unique()
    prev_iid, prev_sid = -1, -1
    pos = 0
    #import code; code.interact(local=dict(globals(), **locals()))
    for i in tqdm(range(len(test_data)), total=len(test_data)):
        sid = test_data[session_key].values[i]
        iid = test_data[item_key].values[i]
        ts = test_data[time_key].values[i]
        if prev_sid != sid:
            #print(len(pr.term_index), len(pr.doc_index))
            prev_sid = sid
            pos = 0
        else:
            if items is not None:
                if np.in1d(iid, items):
                    items_to_predict = items
                else:
                    items_to_predict = np.hstack(([iid], items))

            crs = time.clock()
            trs = time.time()

            for m in metrics:
                if hasattr(m, 'start_predict'):
                    m.start_predict(pr)

            preds = pr.predict_next(sid, prev_iid, items_to_predict, timestamp=ts)

            for m in metrics:
                if hasattr(m, 'stop_predict'):
                    m.stop_predict(pr)

            preds[np.isnan(preds)] = 0
#             preds += 1e-8 * np.random.rand(len(preds)) #Breaking up ties
            preds.sort_values(ascending=False, inplace=True)
            #import code; code.interact(local=dict(globals(), **locals()))

            time_sum_clock += time.clock()-crs
            time_sum += time.time()-trs
            time_count += 1

            for m in metrics:
                if hasattr(m, 'add'):
                    m.add( preds, iid, for_item=prev_iid, session=sid, position=pos )

            pos += 1

        prev_iid = iid

        count += 1


    res = []
    for m in metrics:
        if type(m).__name__ == 'Time_usage_testing':
            res.append(m.result_second(time_sum_clock/time_count))
            res.append(m.result_cpu(time_sum_clock / time_count))
        else:
            res.append( m.result() )

    #import code; code.interact(local=dict(globals(), **locals()))
    print_results(res)
    return res


def print_results(res):
    '''
    Print the result array
        --------
        res : dict
            Dictionary of all results res[algorithm_key][metric_key]
    '''
    #import code; code.interact(local=dict(globals(), **locals()))
    for el in res:
        print(el[0], ': ', round(el[1], 3))

