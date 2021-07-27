import numpy as np
import pandas as pd


def ranked_results(predict_for_item_ids, scores, normalize):
    predictions = np.zeros(len(predict_for_item_ids))
    mask = np.in1d( predict_for_item_ids, list(scores.keys()) )
    items = predict_for_item_ids[mask]
    values = [scores[x] for x in items]
    predictions[mask] = values
    series = pd.Series(data=predictions, index=predict_for_item_ids)
    series.fillna(0)
    if normalize:
        series = series / series.max()
    return series


def cos_similarity(query_bow, cand_bow):
    term_intersec = set(query_bow.keys()) & set(cand_bow.keys())
    v_a = np.array([query_bow[i] for i in term_intersec])
    v_b = np.array([cand_bow[i] for i in term_intersec])
    dot_product = (v_a * v_b).sum()

    v_A = np.fromiter(query_bow.values(), dtype=float)
    v_B = np.fromiter(cand_bow.values(), dtype=float)
    mags = np.sqrt((v_A ** 2).sum()) * np.sqrt((v_B ** 2).sum())
    return 2 * dot_product / mags


def binary(first, second):
    a = len(first & second)
    b = len(first)
    c = len(second)
    result = (2 * a) / ((2 * a) + b + c)
    return result


def random(first, second):
    return random.random()
