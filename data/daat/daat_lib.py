# Data Analysis and Augmentation Toolkit (DAAT) - main file
# Developed at TH Rosenheim  
# &copy; 2020/21 Dominik Stecher, M.Sc.; TH Rosenheim  
# &copy; 2021/22 Florian Bayeff-Filloff, M.Sc.; TH Rosenheim  

# Bibliotheken
import sys
import math
import random
import collections
import warnings

import numpy as np
import pandas as pd
import seaborn as sns

# Fortschrittsbalken
from tqdm.notebook import trange, tqdm

# matplotlib
import matplotlib.pyplot as plt

# scipy
import scipy.interpolate as spi

# sklearn
import sklearn.svm as skl_svm
import sklearn.metrics as skl_m
import sklearn.preprocessing as skl_prep

from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline

# Warnungen
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

# Eigene Funktionen, Klassen, Algorithmen, etc. in .py Dateien
from .data import daat_helper
from .data import THRO_3D_Clustering as th_cluster

__all__ = ['Gen_Distribution', 'Gen_Spline', 'Gen_Recombine', 'Gen_Cluster', 
            'Gen_NextMean', 'Gen_None', 'Instruction', 'Analyser', 'Generator', 
            'Verification']

# ============================================================================ #
# Klassendefinition: Sample_Generator
# ============================================================================ #
class Sample_Generator:
    '''Generatormethoden Grundklasse'''
    
    def __init__(self, rng_min = float('-inf'), rng_max = float('inf')):
        '''     
        Parameter:
        ----------
        rng_min - Float : Untergrenze der Featurewerte
        rng_max - Float : Obergrenze der Featurewerte
        
        Rückgabewert:
        ----------
        Generator Objekt
        '''
        self.value_range = [0.0, 0.0]
        self.value_range[0] = rng_min
        self.value_range[1] = rng_max

        
    def run(self, org_data, syn_data, n_samples, f_id, f_dep_id):
        '''Ausführende Funktion, implementiert Generatormethode und wird vom 
        daat Generator aufgerufen.
        
        Parameter:
        ----------
        org_data - Numpy Array; Original Daten als Numpy Array
        syn_data - Numpy Array; künstliche Daten als Numpy Array
        n_samples - Integer; Anzahl der zu erzeugenden Samples
        f_id - Integer; Idex des zu erzeugenden Features
        f_dep_id - Integer; Indeces aller zu beachtenden Features
        
        Rückgabewert:
        ----------
        Numpy Array mit erzeugten Samples
        '''
        return 0
        
        ''' Verbesserung: 
        nur noch org_data, syn_data -> f_id, f_dep_id unnötig
        Index 0 = zu generierendes Feature
        Index 1+ = zu beachtende Featurewerte
        '''
        
    
    def get_val_rng(self):
        '''Gibt gültigen Wertebereich zurück.
        '''
        val_min = self.value_range[0]
        val_max = self.value_range[1]
        return val_min, val_max
    
    def check_vals_vs_rng(self, vals, rng_min, rng_max):
        '''Kontrolliere erzuegte Feature Werte gegen Wertbereichgrenzen.'''
        for i in range (0, len(vals)):
            if vals[i] > rng_max: vals[i] = rng_max
            if vals[i] < rng_min: vals[i] = rng_min
        return vals
    
# ---------------------------------------------------------------------------- #
class Gen_Distribution(Sample_Generator):
    '''Generatormethode - erzeugt Featurewerte über original Distribution'''
    def __init__(self, rng_min = float('-inf'), rng_max = float('inf'), dist_typ='normal'):
        '''Erstellt ein neues Generator Objekt für Distributionsberechnung.
        
        Parameter:
        ----------
        rng_min - Float : Untergrenze der Featurewerte
        rng_max - Float : Obergrenze der Featurewerte
        dist_typ - String : Name der Distributionsart, wahlweise normal oder
            uniform; Default: normal
        
        Rückgabewert:
        ----------
        Generator Klassen Objekt
        '''
        super().__init__(rng_min, rng_max)
        self.__typ = dist_typ
    
    def run(self, org_data, syn_data, n_samples, f_id, f_dep_ids):
        result = np.zeros((n_samples))
        f_val = org_data[:, f_id]
        f_min, f_max = self.get_val_rng()
        
        gen = np.random.default_rng()
        
        # Werte nach feature filtern und mean & std bestimmen
        f_mean = np.mean(f_val)
        f_std = np.std(f_val)

        # bestimme für jedes Sample einen Wert innerhalb des vorhandenen Wertebereichs
        if self.__typ == 'normal':
            result = gen.normal(f_mean, f_std, n_samples)
        elif self.__typ == 'uniform': 
            result = gen.uniform(f_min, f_max, n_samples)
        else:
            print('Distributions Typ nicht unterstützt')
            
        result = self.check_vals_vs_rng(result, f_min, f_max)
        return result
    
# ---------------------------------------------------------------------------- #
class Gen_Spline(Sample_Generator):
    def __init__(self, rng_min = float('-inf'), rng_max = float('inf')):
        '''Erstellt ein neues Generator Objekt für Spline Berechnung.
        
        Parameter:
        ----------
        rng_min - Float : Untergrenze der Featurewerte
        rng_max - Float : Obergrenze der Featurewerte
        
        Rückgabewert:
        ----------
        Generator Klassen Objekt
        '''
        super().__init__(rng_min, rng_max)
        
    def run(self, org_data, syn_data, n_samples, f_id, f_dep_id):
        result = np.zeros((n_samples))
        f_val = org_data[:, f_id]
        f_min, f_max = self.get_val_rng()
        
        points = org_data[:, f_dep_id]
        new_points = syn_data[:,f_dep_id]

        new_values = spi.griddata(points, f_val, new_points, method='nearest')
        new_values[np.isnan(new_values)] = 0
        
        result = self.check_vals_vs_rng(new_values, f_min, f_max)
        return result

# ---------------------------------------------------------------------------- #
class Gen_Recombine(Sample_Generator):
    def __init__(self, knn_value, rng_min = float('-inf'), rng_max = float('inf')):
        '''Erstellt ein neues Generator Objekt.
        
        Parameter:
        ----------
        knn_value - Integer : Anzahl gesuchte Knn
        rng_min - Float : Untergrenze der Featurewerte
        rng_max - Float : Obergrenze der Featurewerte
        
        Rückgabewert:
        ----------
        Generator Klassen Objekt
        '''
        super().__init__(rng_min, rng_max)
        self.__knn = int(knn_value)
        
    def run(self, org_data, syn_data, n_samples, f_id, f_dep_id):
        result = np.zeros((n_samples))
        f_val = org_data[:, f_id]
        f_min, f_max = self.get_val_rng()
        
        # original data
        o_data = org_data
        s_data = syn_data  
        syn_null = np.all((syn_data == 0))
        
        
        if f_dep_id is not None:      
            if type(f_dep_id) != np.ndarray : f_dep_id = [f_dep_id]
            org_data = o_data[:, f_dep_id]
            gen_data = s_data[:, f_dep_id]
            
        if syn_null:
            # X Zufalls Samples auswählen aus Originaldaten
            rand_ids = np.random.choice(len(data), n_samples, replace=False)
            for i in range (0, n_samples):
                result[i] = o_data[rand_ids[i], f_id]
        else:
            # knn_value näheste Samples aus Originaldaten bestimmen und zufällig Wert auswählen
            if self.__knn > len(o_data): self.__knn = len(o_data)
            dist, ids = daat_helper.knn(s_data, o_data, self.__knn)
            
            for i in range (0, n_samples):
                vectors = np.zeros((self.__knn, o_data.shape[1]))
                for j in range(0, (self.__knn)): 
                    vectors[j] = o_data[ids[i,j]]
                    
                vectors = np.reshape(vectors, (self.__knn, o_data.shape[1]))
                rc = np.random.randint(0, self.__knn, o_data.shape[1])    
                result[i] = vectors[rc[f_id], f_id]
                
        result = self.check_vals_vs_rng(result, f_min, f_max)
        return result

# ---------------------------------------------------------------------------- #
class Gen_Cluster(Sample_Generator):
    def __init__(self, cluster, rng_min = float('-inf'), rng_max = float('inf'),
                 cluster_bound = False):
        '''Erstellt ein neues Generator Objekt .
        
        Parameter:
        ----------
        cluster - Integer : Anzahl gesuchte Cluster
        rng_min - Float : Untergrenze der Featurewerte
        rng_max - Float : Obergrenze der Featurewerte
        
        Rückgabewert:
        ----------
        Generator Klassen Objekt
        '''
        super().__init__(rng_min, rng_max)
        self.__n_cluster = cluster
        self.__c_bound = cluster_bound
        
    def run(self, org_data, syn_data, n_samples, f_id, f_dep_id):
        result = np.zeros((n_samples))
        f_val = org_data[:, f_id]
        f_min, f_max = self.get_val_rng()
        
        # bestimme Feature IDs fuer Vorverarbeitungsfilter
        feature_keys = f_id
        if f_dep_id is not None:
            feature_keys = np.append(feature_keys, f_dep_id)
        else:
            feature_keys = [f_id]
        
        # cluster original daten
        data_cluster_org = org_data[:, feature_keys]
        data_cluster_syn = syn_data[:, feature_keys]
        
        # Cluster bestimmen
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            found_cluster = th_cluster.static_clustering_2(data_cluster_org, self.__n_cluster, verbose = 0)
            uniq_cluster = np.unique(found_cluster)
            n_cluster = len(uniq_cluster)
        
        # Arrays für Cluster Informationen
        w_cluster = len(feature_keys)
        cluster_mean = np.zeros((n_cluster, w_cluster))
        cluster_std = np.zeros((n_cluster, w_cluster))
        cluster_min = np.zeros((n_cluster, w_cluster))
        cluster_max = np.zeros((n_cluster, w_cluster))
       
        # filter Daten nach Cluster und bestimme mean & std
        for i in range(0, n_cluster):
            cluster_id = uniq_cluster[i]
            # ordne gefilterte original Daten gefunden Cluster zu
            pure_cluster = data_cluster_org[np.where(found_cluster == cluster_id)]
            
            cluster_mean[i] = np.mean(pure_cluster)
            cluster_std[i] = np.std(pure_cluster)
            cluster_min[i] = np.min(pure_cluster)
            cluster_max[i] = np.max(pure_cluster)
           
        
        distances = np.zeros((len(data_cluster_syn), n_cluster))
        
        # bestimme nähestes Cluster
        for j in range(0, len(data_cluster_syn)):
            for k in range(0, n_cluster): 
                distances[j][k] = np.linalg.norm(data_cluster_syn[j] - cluster_mean[k])

        # _, min_id = daat_helper.knn(data_cluster_syn, cluster_mean, 1) 
        
        # if self.__c_bound:
        #    c_min = cluster_min[min_id, 0]
        #    c_max = cluster_max[min_id, 0]
        # else:
        
        c_min = f_min
        c_max = f_max
        
        # Clusterwerte für jedes synthetische Sample bestimmen
        for i in range(0, len(result)):
            min_cl = np.argmin(distances[i])
            r_val = np.random.normal(cluster_mean[min_cl, 0], cluster_std[min_cl, 0])
            # r_val = np.random.normal(cluster_mean[min_id[i], 0], cluster_std[min_id[i], 0])
            while r_val > c_max or r_val < c_min:
                # r_val = np.random.normal(cluster_mean[min_id[i], 0], cluster_std[min_id[i], 0])
                r_val = np.random.normal(cluster_mean[min_cl, 0], cluster_std[min_cl, 0])
            result[i] = r_val
            
        result = self.check_vals_vs_rng(result, f_min, f_max)
        return result
    
# ---------------------------------------------------------------------------- #
class Gen_NextMean(Sample_Generator):
    def __init__(self, knn_value, rng_min = float('-inf'), rng_max = float('inf')):
        '''Erstellt ein neues Generator Objekt .
        
        Parameter:
        ----------
        knn_value - Integer : Anzahl gesuchte knn
        rng_min - Float : Untergrenze der Featurewerte
        rng_max - Float : Obergrenze der Featurewerte
        
        Rückgabewert:
        ----------
        Generator Klassen Objekt
        '''
        super().__init__(rng_min, rng_max)
        self.__knn = knn_value
        
    def run(self, org_data, syn_data, n_samples, f_id, f_dep_id):
        result = np.zeros((n_samples))
        f_val = org_data[:, f_id]
        f_min, f_max = self.get_val_rng()
        
        o_data = org_data
        s_data = syn_data
        
        if self.__knn > len(o_data): self.__knn = len(o_data)
        if self.__knn < 2: self.__knn = 2
        
        if f_dep_id is not None:
            if type(f_dep_id) != np.ndarray : f_dep_id = [f_dep_id]
            o_data = org_data[:, f_dep_id]
            s_data = syn_data[:, f_dep_id]
        
        dist, ids = daat_helper.knn(s_data, o_data, self.__knn)
        
        for i in range (0, n_samples):
            val = 0
            for j in range(0, self.__knn):
                val = val + org_data[ids[i,j]][f_id]
            result[i] = val / self.__knn
        
        result = self.check_vals_vs_rng(result, f_min, f_max)
        return result

# ---------------------------------------------------------------------------- #
class Gen_None(Sample_Generator):
    def __init__(self, rng_min = float('-inf'), rng_max = float('inf')):
        '''Erstellt ein neues Platzhalter Generator Objekt .
        
        Parameter:
        ----------
        rng_min - Float : Untergrenze der Featurewerte
        rng_max - Float : Obergrenze der Featurewerte
        
        Rückgabewert:
        ----------
        Generator Klassen Objekt
        '''
        super().__init__(rng_min, rng_max)
        
    def run(self, org_data, syn_data, n_samples, f_id, f_dep_id):
        result = np.zeros((n_samples))
        f_val = org_data[:, f_id]
        f_min, f_max = self.get_val_rng()
        return result

# ============================================================================ #
# Klassendefinition: Instruktion
# ============================================================================ #
class Instruction:
    def __init__(self, feature, generator:Sample_Generator, feature_dep = None):
        self.__feature = feature
        self.__generator = generator
        if feature_dep is not None and type(feature_dep) is not list:
            feature_dep = [feature_dep]
        self.__feature_dep = feature_dep
        
    def get_feature(self):
        return self.__feature
        
    def get_generator(self):
        return self.__generator
        
    def get_feature_dep(self):
        return self.__feature_dep


# ============================================================================ #
# Klassendefinition: DAAT Analyser
# ============================================================================ #
class Analyser:
    '''
    Klasse zur Analyse von Datensätzen.
    ...
    Attribute:
    ----------
    dataset - Pandas Dataframe, Datensatz zur Analyse und Bearbeitung
    
    Methoden:
    ----------
    - set_target_label(target_label)
        Setzt Ziellabelwert
    - show_dataset_info()
        Zeigt Datensatzinformationen an
    - show_feature_info(feature)
        Zeigt Featureinformationen an
    - plot_features(f1, f2, typ, inline, color)
        Erstellt Graphenübersicht für Features in f1
    - plot_pair()
        Erstellt Pair Plot über alle Feature
    - plot_correlation(annot)
        Erstellt Korrelations Plot über alle Feature
    - drop_feature(feature)
        Entfernt Feature aus Datensatz
    - drop_line(line_index)
        Entfernt Eintrag aus Datensatz
    - drop_nan(feature)
        Entfernt Leereinträge nach Feature aus Datensatz
    - fill_nan(feature, value)
        Ersetzt Leereinträge von feature mit value 
    - rename_feature(feature, new_name)
        Benennt feature um in new_name
    - show_outlier(feature, n_percent)
        Bestimmt Ausreißer zu feature über Winsorizing und zeigt an
    - remove_outlier()
        Entfernt bestimmte Ausreißer aus Datensatz 
    '''
    
    __SIZE = 5.0
    
    def __init__(self, dataset, target_label=None):
        '''
        Erstellt ein neues Analyser Objekt, initialisiert mit dem Datensatz
        dataset und dem Ziellabelnamen target_label.
        
        Parameter:
        ----------
        dataset - Pandas Dataframe, enthält Datensatz zur Analyse
        target_label - String, Name des Ziellabels, Default: None
        
        Rückgabewert:
        ----------
        Analyser Klassen Objekt
        '''
        self.dataset = dataset
        
        self.__target = target_label
        self.__thrs = len(dataset) * 0.1
        self.__outlier_ids = None
        
    # ------------------------------------------------------------------------ #
    # Setter
    # ------------------------------------------------------------------------ #
    def set_target(self, target_label):
        '''Setzt Zielwert auf target_label'''
        self.__target = target_label

    # ------------------------------------------------------------------------ #        
    # Informationsanzeige
    # ------------------------------------------------------------------------ #
    def show_dataset_info(self):
        '''zeigt allgemeine Informationen zum Datensatz in Tabellenform an.'''
        line_len = 112
        
        df_nan = self.dataset[self.dataset.isna().any(axis=1)]
        data_col = self.dataset.columns
        
        nr_feature = len(data_col)
        nr_entries = len(self.dataset)
        nr_nan = len(df_nan)
        
        f_uniq = np.zeros(0)
        
        for i in trange(nr_feature, desc='processing features', leave=False):
            feature_data = self.dataset.iloc[:,i]
            uniq = len(pd.unique(feature_data))
            f_uniq = np.append(f_uniq, [uniq])
            
        npy_min = np.argmin(f_uniq)
        uni_min = [data_col[npy_min], int(f_uniq[npy_min]), 'Regression' ]
        npy_max = np.argmax(f_uniq)
        uni_max = [data_col[npy_max], int(f_uniq[npy_max]), 'Regression' ]

        if uni_min[1] < self.__thrs: uni_min[2] = 'Klassifikation'
        if uni_max[1] < self.__thrs: uni_max[2] = 'Klassifikation'
            
        data_des = self.dataset.describe()
        
        print('Datensatz Übersicht')
        print('=' * line_len)
        print('Feature Anzahl:  {:10}'.format(nr_feature) )
        print('Anazhl Einträge: {:10}'.format(nr_entries) )
        print('leere Einträge:  {:10}'.format(nr_nan) )
        print('-' * line_len)
        print('kleinstes Feature : {:10} | Anzahl {:10} | ML Art: {:10}'
              .format(str(uni_min[0]), str(uni_min[1]), uni_min[2]) )
        print('größtes Feature   : {:10} | Anzahl {:10} | ML Art: {:10}'
              .format(str(uni_max[0]), str(uni_max[1]), uni_max[2]) )
        print('')
        
        tab_line_head = '{:20} | {:10} | {:10} | {:10} | {:10} || {:10} | {:10} | {:10} '
        tab_line_1 = '{:20} | {:10} | {:10d} | {:10d} | {:10d} || {:10} | {:10} | {:10} '
        tab_line_2 = '{:20} | {:10} | {:10d} | {:10d} | {:10d} || {:10.2f} | {:10.2f} | {:10.2f} '
        tab_col = ['Feature', 'Type', 'Gesamt', 'Einzel', 'NaNs', 'min', '50%', 'max']
            
        print('Feature Übersicht')
        print('-' * line_len)
        print(tab_line_head.format(*tab_col))
        print('-' * line_len)
        
        for i in range(0, nr_feature):
            f = data_col[i]
            f_vals = self.dataset.iloc[:,i]
            
            nr_uni = len(pd.unique(f_vals))
            nr_nan = f_vals.isnull().sum()
            nr_tot = len(f_vals) - nr_nan
            
            if f in data_des.columns:
                f_min = data_des[f]['min']
                f_max = data_des[f]['max']
                f_mid = data_des[f]['50%']
                
                print(tab_line_2.format(f, str(f_vals.dtypes), nr_tot, nr_uni, 
                                  nr_nan, f_min, f_mid, f_max))
            else:
                print(tab_line_1.format(f, str(f_vals.dtypes), nr_tot, nr_uni, 
                                  nr_nan, '---', '---', '---'))
        
        print('=' * line_len, '\n')
      
    def show_feature_info(self, feature, violin=False):
        '''
        Zeigt allgemeine Informationen zum Feature als Tabelle und 
        Werteverteilung als Histogramm (numerisch) oder Countplot (kategorisch) 
        an.
        
        Parameter:
        ----------
        feature - String,  Feature Name
        violin  - Boolean, Option für Violin Plot, Default: False
        '''
        nr_cols = 2
        data = self.dataset
        numeric = feature in data.select_dtypes(include='number').columns
        
        if feature not in data.columns:
            print('Feature', feature, 'unbekannt')
            return
        data_des = data.describe()
        
        f = data[feature]
        f_type = f.dtype
        nr_f_uni = len(pd.unique(f))
        nr_f_nan = f.isnull().sum()
        nr_f = len(f)

        typ = 'Regression' if (nr_f_uni > self.__thrs) else 'Klassifikation'
                
        table_line = '{:15} {:10.3f}'
        table_rows = ['mean', 'std', 'min', '25%', '50%', '75%', 'max']
          
        feature_text_array = []
        
        f_txt_1 = '\n'.join(('Feature Übersicht ', 
            '='* 26, 'Typ:' + str(f.dtypes),
            '{:15} {:>10}'.format('Anzahl:', nr_f),
            '{:15} {:>10}'.format('Leer:', nr_f_nan),
            '{:15} {:>10}'.format('Einzel:', nr_f_uni),
            '{:10} {:>15}'.format('ML Art:', typ),
            '='* 26))

        if numeric:
            nr_cols = 3
            for i in range(0, len(table_rows)):
                feature_text_array.append(table_line.format(
                    table_rows[i], data_des[feature][table_rows[i]]))
            f_txt_2 = '\n'.join((feature_text_array))
            f_txt = '\n'.join((f_txt_1, f_txt_2, '='* 26)) 
        else:
            f_txt = f_txt_1

        hue_val = self.__target if self.__target is not None else None
        
        f_t = self.__target
        n_t = len(data[f_t].unique()) if f_t is not None else 0
        if n_t > 10: hue_val = None

        fig, ax = plt.subplots(nrows=1, ncols=nr_cols,  sharex=False, sharey=False, 
                               figsize = (nr_cols*self.__SIZE, self.__SIZE))
        fig.suptitle('Feature: ' + feature, fontsize=20)

        
        ax[0].text(0.05, 0.95, f_txt, transform=ax[0].transAxes, fontsize=12, 
                   fontfamily='monospace', va='top', ha="left" )
        ax[0].axis('off')
            
        if nr_cols > 2: 
            sns.histplot(data=self.dataset, x=feature, hue=hue_val, ax=ax[1])
            ax[1].axvline(data_des[feature]["25%"], ls='--', color='r')
            ax[1].axvline(data_des[feature]["mean"], ls='--', color='r')
            ax[1].axvline(data_des[feature]["75%"], ls='--', color='r')
            ax[1].grid(True)

            y = None if hue_val is None else hue_val
            if violin:
                sns.violinplot(data=self.dataset, orient='h', x=feature, y=y, ax=ax[2])
            else:
                sns.boxplot(data=self.dataset, orient='h', x=feature, y=y, ax=ax[2])
            ax[2].grid(True)
        else:
            sns.countplot(data=self.dataset, x=feature, hue=hue_val, ax=ax[1])
            ax[1].grid(True)
        plt.show()
    
    # ------------------------------------------------------------------------ #        
    # Visualisierung
    # ------------------------------------------------------------------------ #
    def plot_features(self, x_feature = None, y_feature = None, typ = None, 
        inline = 3, color = False):
        ''' Generiert grafische Übersicht für alle numerische Feature.
        
        Parameter:
        ----------
        x_feature - String (Liste) : Anzuzeigende Features, Default: None -> alle
        y_feature - String: Feature auf Y-Achse, Default: None
        typ - String: Graph Art, Default: None -> Histogramm
            Ist y_feature gesetzt: Scatterplot
        inline - Integer : Anzahl der Graphen in einer Zeile, Default: 3
        color - Boolean : Wenn target Wert gesetzt: Färbt Graph ein, 
            Default: False
        '''
        mono_plots = {'box' : sns.boxplot, 'hist' : sns.histplot, 
                      'strip' : sns.stripplot}
        multi_plots = {'box' : sns.boxplot, 'scatter': sns.scatterplot, 
                       'line' : sns.lineplot, 'strip' : sns.stripplot}
        
        if x_feature is None:
            x_feature = self.dataset.select_dtypes(include='number').columns
        if type(x_feature) == str: x_feature = [x_feature]
        
        f2_vals = self.dataset[y_feature].to_numpy() if y_feature is not None else None
            
        plot = None
        
        plot_typ = 'hist' if y_feature is None else 'scatter'
        if typ is not None: plot_typ = typ
            
        mono_plot = True if y_feature is None else False

        x_plot = int(inline)
        y_plot_x = int(len(x_feature)/inline +1)
    
        # check number of feature and typ parameters
        if plot_typ in mono_plots:
            plot = mono_plots[plot_typ]
        elif plot_typ in multi_plots:
            plot = multi_plots[plot_typ]
        else:
            print('Plot Typ nicht unterstützt!')
            if mono_plot:
                plot_list = mono_plots.keys()
            else:
                plot_list = multi_plots.keys()
            print('Unterstütze Typen:', list(plot_list) ) 
            return
        
        f_t = self.__target
        n_t = 0
        if f_t is not None:
            t_uniq = self.dataset[f_t].unique()
            n_t = len(t_uniq)
        if f_t is not None and n_t <= 10 and color: 
            hue_val = f_t
        else: 
            hue_val = None
        
        # plot features  
        plt.figure(figsize = (self.__SIZE * x_plot, self.__SIZE * y_plot_x))
        
        for i in trange (len(x_feature), desc='plotting data', leave=False):
            plt.subplot(y_plot_x, x_plot, i+1)
            f1_vals = self.dataset.loc[:,[x_feature[i]]].to_numpy()
            x_vals = f1_vals if y_feature is None else f2_vals
            y_vals = f2_vals if y_feature is None else f1_vals
            
            # plot distribution als line other histogram
            if typ == 'hist':
                ax = plot(data=self.dataset, x=x_feature[i], hue=hue_val)
            else:
                ax = plot(data=self.dataset, x=x_feature[i], y=y_feature, hue=hue_val)
            ax.set(xlabel=x_feature[i])
            ax.xaxis.set_major_locator(plt.AutoLocator())
            ax.yaxis.set_major_locator(plt.AutoLocator())
            
        plt.tight_layout()
        plt.show()
    
    def plot_pair(self):
        '''Zeigt Pairplot aller numerischer Feature an.'''
        numeric_features = self.dataset.select_dtypes(include='number').copy()
        grid = sns.PairGrid(numeric_features)
        grid.map_diag(sns.histplot)
        grid.map_offdiag(sns.scatterplot)
        plt.tight_layout()
        plt.show()
        
    def plot_correlation(self, annot = False):
        '''
        Zeigt Korrelationsmatrix aller numerischer Feature an.
        
        Parameter:
        ----------
        annot - Boolean : Zeigt Werte im Graphen an, Default: False
        '''
        corr = self.dataset.corr()
        fig = plt.figure(figsize=(3 * self.__SIZE, 3 * self.__SIZE))
        sns.heatmap(corr, square=True, annot=annot, vmin = -1, center = 0, 
                    vmax = 1, cbar_kws={"shrink": .75})
        plt.title('Correlation')
        plt.show()
    
    # ------------------------------------------------------------------------ #        
    # Bearbeitung
    # ------------------------------------------------------------------------ #
    def transform_feature(self, feature):
        '''
        Transformiert kategorisches Feature in numerisches und gibt 
        LabelEncoder Objekt zurück.
        
        Parameter:
        ----------
        feature - String : Zu transformierendes Feature
        '''
        encoder = None
        
        if feature in self.dataset.columns :
            encoder = skl_prep.LabelEncoder()
            encoder = encoder.fit(self.dataset[feature])
            self.dataset[feature] = encoder.transform(self.dataset[feature])
        else:
            print('Feature unbekannt')
            
        return encoder
 
    def drop_feature(self, feature):
        '''
        Entfernt feature.
        
        Parameter:
        ----------
        feature - String (Array): Name der Feature
        '''
        if type(feature) == str: feature = [feature]
        self.dataset = self.dataset.drop(feature, axis = 1)
    
    def drop_line(self, line):
        '''
        Entfernt Einträge mit Indices line.
        
        Parameter:
        ----------
        line - Integer (Array) : Indexes der Einträge
        '''
        self.dataset = self.dataset.drop(line, axis = 0)
        self.dataset = self.dataset.reset_index(drop = True)
    
    def drop_nan(self, feature):
        '''
        Entfernt Leereinträge nach feature gefiltert.
        
        Parameter:
        ----------
        feature - String (Array): Name der Feature
        '''
        nan_lines = self.dataset[self.dataset[feature].isna()].index
        self.drop_line(nan_lines)
        print('removed', len(nan_lines), 'NaN Entries')
    
    def fill_nan(self, feature, value=None):
        ''' 
        Ersetzt Leerfelder nach feature gefiltert mit value, ansonsten mit 
        Median.
        
        Parameter:
        ----------
        feature - String (Array) : Name der Feature
        value - Numerisch : Wert zum Ersetzen
        '''
        nan_lines = self.dataset[self.dataset[feature].isna()].index
        if value is None:
            self.dataset[feature].fillna(self.dataset[feature].median(), inplace=True)
            print('filled', len(nan_lines), 'NaN Entries with Median')
        else:
            self.dataset[feature].fillna(value, inplace=True)
            print('filled', len(nan_lines), 'NaN Entries with', value)
    
    def rename_feature(self, feature, new_name):
        '''
        Benennt feature in new_name um.
        
        Parameter:
        ----------
        feature - String : Feature Name
        name -    String : neuer Name
        '''
        self.dataset.rename(columns = {feature:new_name}, inplace = True)
    
    def show_outlier(self, feature, n_percent = 2, inline = 3):
        '''
        Erkennt Ausreißer von feature über Winserize n_percent und zeigt sie 
        farblich markiert im Histogramm aller Feature an.
        
        Parameter:
        ----------
        feature - String : Name der Feature
        n_percent - Float : Äußerer Prozentbereich
        inline - Integer : Anzahl der Graphen in einer Zeile
        '''
        tmp_data = self.dataset.copy()
        nr_enties = len(tmp_data)
        
        data_feature = tmp_data[feature]
        border = (n_percent/2.0)/100.0
        p1 = data_feature.quantile(border)
        p2 = data_feature.quantile(1 - border)
        
        self.__outlier_ids = data_feature.loc[(data_feature < p1) | (data_feature > p2)].index
        
        tmp_data['outlier'] = False
        tmp_data.loc[self.__outlier_ids, 'outlier'] = True
        
        x_plot = int(inline)
        y_plot = int(len(self.dataset.columns)/inline +1)
        
        plt.figure(figsize = (self.__SIZE * x_plot, self.__SIZE * y_plot))
        
        for i in range (0, len(self.dataset.columns)):
            plt.subplot(y_plot, x_plot, i+1)
            ax = sns.histplot(tmp_data, x=self.dataset.columns[i], hue='outlier', multiple='stack')
            ax.set(xlabel=self.dataset.columns[i])
        plt.show()
    
    def remove_outlier(self):
        '''Entfernt alle mit show_outlier() markierten Einträge.'''
        if self.__outlier_ids is None:
            print('noch keine Ausreißer definiert. Zuerst show_outlier() ausführen.')
        else:
            print('entferne', len(self.__outlier_ids), 'Ausreißer')
            self.drop_line(self.__outlier_ids)
            self.__outlier_ids = None

# ============================================================================ #
# Klassendefinition: DAAT Generator / Augmentor
# ============================================================================ #
class Generator: 
    '''
    Klasse zum Generieren virtueller Samples
    ...  
    Methoden:
    ----------
    - add_instruction(instruction)
    - remove_instruction(feature)
    - status()
    - verify_setup(classification, balance)
    - generate_syn_data(self, n_samples, classification, balance, equal)
    - get_syn_data(combine)
    '''
    __max_iter = 1e3
    __large_num = 1e200 # riesige Nummer, die ziemlich sicher nicht vorkommt
    
    def __init__(self, data, target):
        '''
        Erstellt ein neues Generator Objekt, initialisiert mit dem Datensatz
        dataset und dem Ziellabelnamen target_label.
        
        Parameter:
        ----------
        dataset - Pandas Dataframe, enthält Datensatz zur Analyse
        target_label - String, Name des Ziellabels, Default: None
        
        Rückgabewert:
        ----------
        Generator Klassen Objekt
        '''
        
        if target not in data.columns:
            print('Ziellabel nicht in Datensatz vorhanden.')
            return
        
        # original Datensatz
        tmp_data = data.drop(target, axis=1)
        self.__data_samples = tmp_data.to_numpy()
        self.__data_labels = data[target].to_numpy()
        self.__data_count = len(self.__data_samples)
        
        # 'target' Spalte muss als letztes stehen -> Indecies
        self.__data_col = np.append(tmp_data.columns, target)
        self.__target = target
        
        # synthetischer Datensatz
        self.__data_syn_samples = None
        self.__data_syn_labels = None
        self.__data_syn_count = 0
        
        # Liste der Instruktionen als Dictonary
        self.__instructions = dict()
    
    # ------------------------------------------------------------------------ #        
    # Generator Pipeline Definition
    # ------------------------------------------------------------------------ #
    def add_instruction(self, instruction:Instruction):
        ''' 
        Fügt instruction zu Erzeugungspipeline hinzu. 
        Für jedes Feature außer dem Target muss ein Eintrag vorhanden sein.
        
        Parameter:
        ----------
        instruction - Instruction : Anweisungs Objekt für Erzeugungspipeline
        '''
        control = True
        feature_dep = instruction.get_feature_dep()
        feature = instruction.get_feature()
        
        if feature not in self.__data_col:
            print('Feature', feature, 'nicht in Datensatz vorhanden.')
            control = False
        if feature_dep is not None:
            for i in range(0, len(feature_dep)):
                if feature_dep[i] not in self.__data_col:
                    print('Feature', feature_dep[i], 'nicht in Datensatz vorhanden.')
                    control = False
                    
        if control: 
            self.__instructions[feature] = instruction
               
    # ------------------------------------------------------------------------ #
    def remove_instruction(self, feature_label:str) -> bool:
        '''
        Entfernt Instruktion für feature aus Erzeugungspipeline.
        
        Parameter:
        ----------
        feature - String : Name des Features
        '''
        control = True
        
        if feature_label in self.__instructions:
            del self.__instructions[feature_label]
            print('Anweisung für', feature_label, 'entfernt.')
        else:
            print('[ERROR] Anweisung für', feature_label, 'nicht vorhanden.')
            control = False
            
        return control
      
    # ------------------------------------------------------------------------ #
    def status(self):
        '''Zeigt aktuell definierte Erzeugungspipeline an.'''
        ins = self.__instructions
        ins_list = list(ins)
        n_ins = len(ins_list)
        
        line = '   {:10} : {:30}'
        print('Aktuell', n_ins, 'Instruktion(en) enthalten.')
        
        for i in range(0, n_ins):
            
            print('Index', i)
            cur_ins = ins[ins_list[i]]
            print(line.format('Feature', str(cur_ins.get_feature())))
            gen_name = type(cur_ins.get_generator()).__name__
            print(line.format('Generator', gen_name))
            print(line.format('beachtetet', str(cur_ins.get_feature_dep())))
    
    # ------------------------------------------------------------------------ #        
    # Virtuelle Datensatz Erzeugung
    # ------------------------------------------------------------------------ #
    @ignore_warnings(category = ConvergenceWarning)
    def verify_setup(self, classification = True, balance = False):
        '''Erzeugt virtuellen Datensatz nach definierter Erzeugungspipeline. 
        Zeigt Vergleichsübersicht für zwei SVMs trainiert auf original und 
        erzeugten Datensatz an. SVMs begrenzt auf 1000 Durchläufe.
        
        Parameter:
        ----------
        classification - Boolean :  SVM Art, default = Klassifikation.
        balance - Boolean : balanziert Klassen bei der Erzeugung aus.
        '''
        # neue syn Samplemenge = Originalmenge erzeugen
        error_code = self.generate_syn_data(self.__data_count, classification, balance)

        # Fehlercode 0: alles ok, mach weiter 
        if error_code == 0:
            x_data = self.__data_samples
            y_data = self.__data_labels
            
            x_syn = self.__data_syn_samples
            y_syn = self.__data_syn_labels
            
            strat_val = [None, None]
            y_labels, label_count = np.unique(y_data, return_counts=True)
            if classification:
                strat_val = [y_data, y_syn]
                if len(label_count > 2):
                    y_data = skl_prep.label_binarize(y_data, classes=y_labels)
                    y_syn = skl_prep.label_binarize(y_syn, classes=y_labels)
                
            X_train_org, X_test_org, y_train_org, y_test_org = train_test_split(
                x_data, y_data, test_size=0.3, random_state=0, stratify=strat_val[0])
            X_train_syn, X_test_syn, y_train_syn, y_test_syn = train_test_split(
                x_syn, y_syn, test_size=0.3, random_state=0, stratify=strat_val[1])
            
            if classification:
                svm_org = make_pipeline(skl_prep.StandardScaler(), OneVsRestClassifier(
                    skl_svm.SVC(kernel ='linear', max_iter=self.__max_iter))).fit(
                    X_train_org, y_train_org)
                svm_syn = make_pipeline(skl_prep.StandardScaler(), OneVsRestClassifier(
                    skl_svm.SVC(kernel='linear',max_iter=self.__max_iter))).fit(
                    X_train_syn, y_train_syn)
            else:
                svm_org = make_pipeline(skl_prep.StandardScaler(), skl_svm.SVR(kernel='linear', 
                    max_iter=self.__max_iter)).fit(X_train_org, y_train_org)
                svm_syn = make_pipeline(skl_prep.StandardScaler(), skl_svm.SVR(kernel='linear',
                    max_iter=self.__max_iter)).fit(X_train_syn, y_train_syn)
            
            org_org_pred = svm_org.predict(X_test_org)
            syn_syn_pred = svm_syn.predict(X_test_syn)
            org_syn_pred = svm_syn.predict(X_test_org)
            
            y_deci = None
            if classification:
                org_org_deci = svm_org.decision_function(X_test_org) 
                syn_syn_deci = svm_syn.decision_function(X_test_syn) 
                org_syn_deci = svm_syn.decision_function(X_test_org)
                y_deci = [org_org_deci, syn_syn_deci, org_syn_deci]
        
            y_test = [y_test_org, y_test_syn, y_test_org]
            y_pred = [org_org_pred, syn_syn_pred, org_syn_pred]

            if classification:
                self.__verify_class(y_test, y_pred, y_deci, y_labels, len(label_count))
            else:
                self.__verify_regression(y_test, y_pred)

        return error_code

    def generate_syn_data(self, n_samples, classification=True, balance = False, equal = False):
        '''
        Erezugt eine beliebige Anzahl synthetische Samples basierend auf den
        Originaldaten.
        
        Parameter:
        ----------
        n_samples - Integer : Anzahl zu erzeugender Samples
        classification - Boolean: 
          
        '''
        e_code = 0
        
        # Klassifikation
        if classification:
            # Initialisierung temp Arrays fuer Generierung
            result_samples = np.zeros((0,len(self.__data_samples[0])))
            result_labels = np.zeros(0)
            class_n_samples = []
            class_n_syn_samples = []
            
            # Klassenverhaeltnisse bestimmen
            classes, n_class_samples = np.unique(self.__data_labels, return_counts=True)
            n_class_ratio = n_class_samples / self.__data_count
            
            # berechne uebrige Sampleanzahl nach balance zur groessten Klasse
            max_class_id = np.argmax(n_class_samples)
            n_class_dif = n_class_samples[max_class_id] - n_class_samples
            n_sum_dif = np.sum(n_class_dif)
            rest = n_samples - n_sum_dif
            
            # berechne Anzahl der zu generierenden Samples
            for i in range(len(classes)):
                n_syn_samples =  int(round( n_samples * n_class_ratio[i] ))
                if equal:    
                    n_syn_samples = int(round( n_samples * ( 1.0 / len(classes) ) ))
                if balance:
                    if rest < 0:
                        # Gesamtmenge an Samples, wenn nur kleiner kleiner Klasse generiert wird - Anzahl vorhanden kleinere Klasse
                        gen_min_samples = (n_class_samples[max_class_id] * (len(classes) - 1)) - (np.sum(n_class_samples) - n_class_samples[max_class_id])
                        print("[WARNING] At least", gen_min_samples, "necessary for balanced sample generation.")
                        return 
                    else:
                        n_syn_samples = int(round(n_class_dif[i] + rest / len(classes)))
                class_n_syn_samples.append(n_syn_samples)
   
            # 5) wiederhole für jede Klasse
            for i in trange(len(classes), desc='processing classes', leave=False):
                # 2) Unterdatensätze für jede Klasse erstellen / filtern
                data_class = np.empty((0, len(self.__data_samples[0])))
                label_class = np.zeros((class_n_syn_samples[i]))
                label_class.fill(classes[i])

                id_class = np.argwhere(self.__data_labels == classes[i]).ravel()
                data_class = self.__data_samples[id_class]
                
                # 3) Erzeuge Syn-Samples
                e_code, syn_class_samples = self.__run(data_class, class_n_syn_samples[i])
                if e_code > 0: return e_code # Fehlermeldung zurückgeben
                
                # 4) tatsächliche Menge überschreiben
                result_samples = np.append(result_samples, syn_class_samples[:class_n_syn_samples[i]], axis=0)
                result_labels = np.append(result_labels, label_class[:class_n_syn_samples[i]])
            
            self.__data_syn_samples = result_samples
            self.__data_syn_labels = result_labels
            self.__data_syn_count = len(result_samples)

        # Regression
        else:
            #self.__data_syn_count = int(n_syn_count)
            #self.__data_syn_samples = np.zeros((int(n_syn_count), len(self.__data_samples[0])))
            
            # 1) erzeuge Syn-Sample aus allen Daten 
            e_code, syn_samples = self.__run(self.__data_samples, int(n_samples))
            if e_code > 0: return e_code

            #n_random_ids = np.random.choice(int(n_syn_count), int(n_samples), replace=False)
            
            self.__data_syn_samples = syn_samples#[n_random_ids]
            self.__data_syn_labels = daat_helper.interpolate_syn_data(
                self.__data_samples, self.__data_labels, self.__data_syn_samples)
            self.__data_syn_count = len(self.__data_syn_samples)
            
        return e_code

    def get_syn_data(self, combine=False):
        ''' Gibt virtuelle erzeugten Datensatz zurück.
        
        Parameter:
        ----------
        combine   bool, default False
          Gibt stattdessen den synthetischen Datensatz kombiniert mit dem
          original zurück.
        '''
        samples = self.__data_syn_samples
        labels = self.__data_syn_labels.reshape(self.__data_syn_count, 1)
        syn_dataset = np.append(samples, labels, axis = 1)
        ds_syn = pd.DataFrame(syn_dataset, columns=self.__data_col)
        
        if combine:
            gen_samples = self.__data_samples
            gen_labels = self.__data_labels.reshape(self.__data_count, 1)
            gen_dataset = np.append(gen_samples, gen_labels, axis = 1)
            ds_gen = pd.DataFrame(gen_dataset, columns=self.__data_col)
            ds_syn = ds_syn.append(ds_gen, ignore_index=True)
            
        return ds_syn
    
    # ------------------------------------------------------------------------ #        
    # Private & Hilfsfunktioenn 
    # ------------------------------------------------------------------------ #
    def __run(self, data, n_samples, DEBUG = False):
        '''Führt Generatorfunktionen aus.'''
        result = np.zeros((n_samples, len(data[0])))
        
        instructions = list(self.__instructions)
        n_instructions = len(instructions)
        
        if n_instructions < len(data[0]):
            col = self.__data_col[self.__data_col != self.__target]
            diff = np.setdiff1d(col, instructions)
            print('Nicht für alle Features eine Instruktion angegeben.')
            print('Es fehlen: ', end = '')
            for i in range(len(diff)): print(diff[i], end = ' ')
            return (1, None)
       
        for i in trange(n_instructions, desc='generating features', leave=False):
            # lese Eintrag aus Dictonary aus
            feature = instructions[i]
            instruction = self.__instructions[feature]
            generator = instruction.get_generator()
            feature_dep = instruction.get_feature_dep()
            
            # Indices Feature und ggf. abhängige Featurewerte bestimmen
            f_index = np.where(self.__data_col == feature)[0][0]
            f_id_dep = None
            if feature_dep is not None:
                f_id_dep = np.where(self.__data_col == feature_dep[0])[0][0]
                for i in range (1, len(feature_dep)):
                    f_id_dep = np.append(f_id_dep, np.where(self.__data_col == feature_dep[i])[0][0])
            
            if generator is not None:
                new_feature = np.zeros((n_samples))
                new_feature = generator.run(data, result, n_samples, f_index, f_id_dep) 
                new_feature = self.__check_integer(data[:, f_index], new_feature)
                result[:,f_index] = new_feature
        return (0, result)
       
    def __verify_regression(self, y_test, y_pred):
        tab_cols = ['Modell', 'R2', 'MSE' ]
        tab_rows = ['Orginal', 'Synthetisch', 'Org. auf Syn.']
        tab_data = np.empty((3,2))

        for i in range(0, 3):
            tab_data[i, 0] = skl_m.r2_score(y_test[i], y_pred[i])
            tab_data[i, 1] = skl_m.mean_squared_error(y_test[i], y_pred[i])
            
        # Tabelle der Werte
        tab_line = '{:15} | {:10} | {:10}'
        cell_text = []
        for row in range(len(tab_data)):
            cell_text.append(['%1.2f' % x for x in tab_data[row]])
        
        print('Auswertung Setup mit SVR')
        self.__print_verify_table(tab_line, tab_rows, tab_cols, cell_text)
        return   
    
    def __verify_class(self, y_test, y_pred, y_deci, y_labels, n_classes):
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        cm = dict()
        
        tab_cols = ['Modell', 'Accuracy', 'Precision', 'Recall', 'F1']
        tab_rows = ['OM auf OD', 'SM auf SD', 'SM auf OD']   
        tab_data = np.empty((3,4))
        cell_text = []
    
        if n_classes > 2:
            y_test_multi = [np.argmax(y_test[0], axis=-1), np.argmax(y_test[1], axis=-1), 
                            np.argmax(y_test[2], axis=-1)]
            y_pred_multi = [np.argmax(y_pred[0], axis=-1), np.argmax(y_pred[1], axis=-1), 
                            np.argmax(y_pred[2], axis=-1)]
            avg_val = 'macro'
        else:
            y_test_multi = y_test
            y_pred_multi = y_pred
            avg_val = 'binary'
        
        for i in range(0,3):
            if n_classes > 2:
                for j in range(n_classes):
                    _id = str(i) + str(j)
                    fpr[_id], tpr[_id], _ = skl_m.roc_curve(y_test[i][:, j], y_deci[i][:, j])
                    roc_auc[_id] = skl_m.auc(fpr[_id], tpr[_id])
                
            fpr[i], tpr[i], _ = skl_m.roc_curve(y_test[i].ravel(), y_deci[i].ravel())
            roc_auc[i] = skl_m.auc(fpr[i], tpr[i])
            cm[i] = skl_m.confusion_matrix(y_test_multi[i], y_pred_multi[i], labels=y_labels)            
            
            tab_data[i, 0] = skl_m.accuracy_score(y_test_multi[i], y_pred_multi[i])
            tab_data[i, 1] = skl_m.precision_score(y_test_multi[i], y_pred_multi[i], zero_division=0, average=avg_val)
            tab_data[i, 2] = skl_m.recall_score(y_test_multi[i], y_pred_multi[i], zero_division=0, average=avg_val)
            tab_data[i, 3] = skl_m.f1_score(y_test_multi[i], y_pred_multi[i], zero_division=0, average=avg_val)
            
            cell_text.append(['%1.2f' % x for x in tab_data[i]])
            
        # Ergebnisanzeige
        fig, ax = plt.subplots(1, 4,  sharex=False, sharey=False, figsize = (20.0, 5.0))

        for i in range (0,3):
            ax[0].plot(fpr[i], tpr[i], label=tab_rows[i] + ' (area = %0.2f)' % roc_auc[i])
        ax[0].plot([0, 1], [0, 1], 'k--')
        ax[0].legend(loc="lower right")
            
        # Confusion Matrices
        for i in range (0, 3):
            ax[i+1].set_title(tab_rows[i])
            skl_m.ConfusionMatrixDisplay(cm[i], display_labels=y_labels).plot(colorbar = False, ax=ax[i+1])
        plt.show()
             
        # Tabelle der Werte
        tab_line = '{:15} | {:10} | {:10} | {:10} | {:10}'

        print('Auswertung Setup mit linearer SVC')
        self.__print_verify_table(tab_line, tab_rows, tab_cols, cell_text)
            
        return 0

    def __print_verify_table(self, tab_line, tab_row, tab_col, tab_text):        
        spaces = len(tab_col) * 10 + (len(tab_col) - 1) * 3
        print('='* spaces)
        
        print(tab_line.format(*tab_col))
        for i in range(0, len(tab_row)):
            print(tab_line.format(tab_row[i], *tab_text[i]))
        print('='* spaces)
        
        return
 
    def __check_integer(self, ref_data, data):
        '''Überprüft Originalfeaturewerte auf möglichen Integertyp, rundet syn.
        Werte entsprechend.'''
        result = data
        if np.all([not (i%1) for i in ref_data]):
            result = np.around(result)
            result = result.astype(int)
        return result

# ============================================================================ #
# Klassendefinition: DAAT Verification
# ============================================================================ #
class Verification:
    def __init__(self, org_data:pd.DataFrame, syn_data:pd.DataFrame):
        # public variables
        self.cache = 200
        self.iterations = 10000
        
        # private variables
        self.__org_data = org_data
        self.__syn_data = syn_data
    
    # ------------------------------------------------------------------------ #    
    # Evaluations Funktionen
    # ------------------------------------------------------------------------ #
    def eval_class_data_set(self, target_label:str, train_ratio:float, weights=None):
        pbar = tqdm(desc = 'fitting SVMs', total = 4)
        
        # Aufteilen in Samples und Label Daten
        org_data_x = self.__org_data.drop(target_label, axis = 1).to_numpy()
        org_data_y = self.__org_data[target_label].to_numpy()
        
        syn_data_x = self.__syn_data.drop(target_label, axis = 1).to_numpy()
        syn_data_y = self.__syn_data[target_label].to_numpy()
        
        # Aufteilen in Trainings und Test Daten
        org_x_train, org_x_test, org_y_train, org_y_test = train_test_split(
            org_data_x, org_data_y, train_size = train_ratio, random_state = 42, 
            stratify = org_data_y)
        
        syn_x_train, syn_x_test, syn_y_train, syn_y_test = train_test_split(
            syn_data_x, syn_data_y, train_size = train_ratio, random_state = 42, 
            stratify = syn_data_y)
        
        # Erstellen der vier Test-SVMs
        svm_ogr = make_pipeline(skl_prep.StandardScaler(), 
            skl_svm.SVC(cache_size = self.cache, max_iter = self.iterations, class_weight = weights))
        svm_synr = make_pipeline(skl_prep.StandardScaler(), 
            skl_svm.SVC(cache_size = self.cache, max_iter = self.iterations, class_weight = weights))
        svm_ogc = make_pipeline(skl_prep.StandardScaler(), 
            skl_svm.SVC(cache_size = self.cache, max_iter = self.iterations, class_weight = weights))
        svm_sync = make_pipeline(skl_prep.StandardScaler(), 
            skl_svm.SVC(cache_size = self.cache, max_iter = self.iterations, class_weight = weights))
        
        # Trainiren der vier Test-SVMs
        svm_ogr.fit(org_x_train, org_y_train)
        pbar.update(1)
        svm_synr.fit(syn_x_train, syn_y_train)
        pbar.update(2)
        svm_ogc.fit(org_data_x, org_data_y)
        pbar.update(3)
        svm_sync.fit(syn_data_x, syn_data_y)
        pbar.update(4)
        pbar.close()
        
        # SVM Test-Werte erstellen
        pred_ogr_o = svm_ogr.predict(org_x_test)
        pred_ogr_s = svm_ogr.predict(syn_x_test)
        pred_synr_o = svm_synr.predict(org_x_test)
        pred_synr_s = svm_synr.predict(syn_x_test)
        pred_ogc_o = svm_ogc.predict(org_data_x)
        pred_ogc_s = svm_ogc.predict(syn_data_x)
        pred_sync_o = svm_sync.predict(org_data_x)
        pred_sync_s = svm_sync.predict(syn_data_x)
        
        # Auswertung
        ogr_o_m = self.__advanced_performance_measures(pred_ogr_o, org_y_test)
        ogr_s_m = self.__advanced_performance_measures(pred_ogr_s, syn_y_test)
        synr_o_m = self.__advanced_performance_measures(pred_synr_o, org_y_test)
        synr_s_m = self.__advanced_performance_measures(pred_synr_s, syn_y_test)
        ogc_o_m = self.__advanced_performance_measures(pred_ogc_o, org_data_y)
        ogc_s_m = self.__advanced_performance_measures(pred_ogc_s, syn_data_y)
        sync_o_m = self.__advanced_performance_measures(pred_sync_o, org_data_y)
        sync_s_m = self.__advanced_performance_measures(pred_sync_s, syn_data_y)
        
        # TABELLE
        tab_line_1 = '| {:10} || {:10.3f} | {:10.3f} || {:10.3f} | {:10.3f} || {:10.3f} | {:10.3f} || {:10.3f} | {:10.3f} |'
        tab_line_2 = '| {:10} || {:10d} | {:10d} || {:10d} | {:10d} || {:10d} | {:10d} || {:10d} | {:10d} |'
        tab_head_1 = '| {:10} || {:23} || {:23} || {:23} || {:23} |'
        tab_head_2 = '| {:10} || {:10} | {:10} || {:10} | {:10} || {:10} | {:10} || {:10} | {:10} |'
        tab_col = ['Metric', 'Accuracy', 'Precision', 'Recall', 'F1', 'TP', 'TN', 'FP', 'FN', 'Sum']
        tab_head_col = ['Modell', '        OG 70', '        Syn 70', '        OG 100', '        Syn 100']
        tab_head_col_2 = ['Metric', '  OG 30', '  Syn 30', '  OG 30', '  Syn 30', '  OG 100', '  Syn 100', '  OG 100', '  Syn 100']
        line_len = 14 + 4 * 17 + 40
        
        print('=' * line_len)
        print('Data Set Evaluation')
        print('-' * line_len)
        print(tab_head_1.format(*tab_head_col))
        print(tab_head_2.format(*tab_head_col_2))
        print('-' * line_len)
        
        for i in range(0, len(ogr_o_m)):
            if i < 4: # ersten vier Werte:
                print(tab_line_1.format(tab_col[i+1], ogr_o_m[i], ogr_s_m[i], synr_o_m[i], 
                    synr_s_m[i], ogc_o_m[i], ogc_s_m[i], sync_o_m[i], sync_s_m[i]))
            else:
                print(tab_line_2.format(tab_col[i+1], ogr_o_m[i], ogr_s_m[i], synr_o_m[i], 
                    synr_s_m[i], ogc_o_m[i], ogc_s_m[i], sync_o_m[i], sync_s_m[i]))
            if i == 3:
                print('-' * line_len)
        
        print('=' * line_len)
        
    # ------------------------------------------------------------------------ #
    # Hilfsfunktionen 
    # ------------------------------------------------------------------------ #    
    def __advanced_performance_measures(self, predicted, truth):

        tp = tn = fp = fn = 0

        for i in range(0, len(predicted)):
            if (predicted[i] == 1 and truth[i] == 1): tp += 1
            elif (predicted[i] == 0 and truth[i] == 0): tn += 1
            elif (predicted[i] == 1 and truth[i] == 0): fp += 1
            elif (predicted[i] == 0 and truth[i] == 1): fn += 1
                
        total = tp + tn + fp + fn
        acc = (tp + tn) / (total)
        
        pos = tp + fp
        tru = tp + fn
        
        pre = 0 if pos == 0 else (tp) / (tp + fp)
        rec = 0 if tru == 0 else (tp) / (tp + fn)
        
        f_s = pre + rec
        
        f1 = 0 if f_s == 0 else 2 * ((pre * rec) / (f_s))

        return [acc, pre, rec, f1, tp, tn, fp, fn, total]
