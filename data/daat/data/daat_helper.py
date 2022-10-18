__all__ = []

import numpy as np
import scipy.interpolate as spi
from sklearn.neighbors import NearestNeighbors
from pandas.api.types import is_numeric_dtype, is_bool_dtype

# ---------------------------------------------------------------------------- #
# Hilfsfunktionen für Berechnungen im DAAT
# ---------------------------------------------------------------------------- #
def euclidian_distance(a, b):
    return np.sqrt(np.sum((a-b)**2, axis=1))

def knn(A, B, k):
    
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(B)
    
    n_dists, n_ids = neigh.kneighbors(A, k)
    return np.array(n_dists), np.array(n_ids)
    
# ---------------------------------------------------------------------------- #
# Hilfsfunktionen für Berechnungen im DAAT
# ---------------------------------------------------------------------------- #
def match_syn_data(X_vals, Y_vals, syn_data):
    syn_results = np.empty((len(syn_data)))
    dist, ids = knn(syn_data, X_vals, 1)

    for i in range (0, len(syn_results)):
        syn_results[i] = Y_vals[ids[i]]
    return syn_results

def interpolate_syn_data(x_data, y_data, x_syn):
    # interpoliere über Originalwerte um Syn-Werte zu bestimmen
    y_syn = spi.griddata(x_data, y_data, x_syn, method='nearest')
    y_syn[np.isnan(y_syn)] = 0.0
    
    if np.all([not (i%1) for i in y_data]): y_syn = y_syn.astype(int)
    return y_syn
    
    
# ---------------------------------------------------------------------------- #
# Hilfsfunktionen zur Datentypen Kontrolle
# ---------------------------------------------------------------------------- #
def check_integer(ref_data, data):
    '''Überprüft Originalfeaturewerte auf möglichen Integertyp, rundet syn.
    Werte entsprechend.'''
    result = data
    if np.all([not (i%1) for i in ref_data]):
        result = np.around(result)
        result = result.astype(int)
    return result
    
    
def check_df_int(df):
    for col in df.columns:
        if is_numeric_dtype(df[col]) and np.array_equal(df[col], df[col].astype(int)):
            df.astype({col : 'int64'}).dtypes
    return df
            