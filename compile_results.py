import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
pd.set_option('precision', 3)
NUM_SLICES = 5


def generate(dataset):
    slugs_df = pd.read_csv('slugs_%s.csv' % dataset, sep=';').set_index('Metrics')
    folder='results/ir/next/%s/final/' % dataset
    files= ['test_window_next_knn_%s.%s.csv', 'test_window_next_knn2_%s.%s.csv',
            'test_window_next_rules_%s.%s.csv', 'test_window_next_neural_%s.%s.csv']
    dfx = None
    df = None
    for j, file in enumerate(files):
        for slice_num in range(NUM_SLICES):
            filename = os.path.join(folder, file % (dataset, slice_num))
            if slice_num == 0 and j == 0:
                df = pd.read_csv(filename, sep=';')
                df['fold'] = slice_num
            else:
                dfx = pd.read_csv(filename, sep=';')[:]
                dfx['fold'] = slice_num
                df = df.append(dfx)
    #skip = ['ir_mc-past_items=%s' % s for s in [0, 1, 2, 3, 5, 6]]
    skip = []
    str1 = ['MRR@%s' % s for s in range(1, 21)]
    str2 = ['HitRate@%s' % s for s in range(1, 21)]
    #import code; code.interact(local=dict(globals(), **locals()))
    #xdf = df[['Metrics', 'fold'] + str1 + str2]
    #xdf = xdf.sort_values('MRR@20: ', ascending=False)
    #xdf.columns = ['metrics', 'hr@1', 'hr@20', 'mrr@1', 'mrr@5', 'mrr@10', 'mrr@20', 'fold']
    #xdf = xdf[~xdf.metrics.isin(skip)]
    file = 'results/%s_condensed_all2.csv' % dataset
    print("Created ", file)
    df.to_csv(file, index=False)

    #metr = df.Metrics.values
    #df = df.groupby(['Metrics']).mean()
    #df = df.reset_index()
    #df.set_index('Metrics', inplace=True)
    #df.insert(0, 'metrics', slugs_df['slug'])
    #df = df[df.metrics != 'skip']
    #df.reset_index(drop=True, inplace=True)


#cl = ['HitRate@%s: ' % x for x in range(1, 21)]
#df_hr = df[['metrics'] + cl].T
#refs = slugs_df.set_index('slug')
#lines = {'neural': '--', 'IR': '-', 'kNN': ':', 'rules': '--'}
#algs = refs[refs.index != 'skip']
#new_header = df_hr.iloc[0]
#df_hr = df_hr[1:]
#df_hr.columns = new_header
#blacklist = ['ir_mc-past_items=1', 'ir_mc-past_items=2', 'ir_mc-past_items=3', 'ir_mc-past_items=5', 'ir_mc-past_items=6']


generate('nowplaying')
generate('30music')
generate('lastfm')
