__all__ = ['split_data', 'run_test_class', 'run_test_reg']

import math
import random

import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib as mpl
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.layers.experimental import preprocessing

from sklearn.model_selection import train_test_split
import sklearn.preprocessing as skl_prep
import sklearn.metrics as skl_m
from sklearn.metrics import ConfusionMatrixDisplay

from collections import Counter

# DAAT Notebook
from .daat_lib import *

def split_data_from_all(data, target, train_ratio, gen_ratio, state = 0, strat = True, verbose = True):
    # Aufteilen in Train, Test, Generate
    data_x = data.drop(target, axis = 1).to_numpy()
    data_y = data[target].to_numpy()

    strat_data = data_y if strat else None
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, train_size=train_ratio, random_state=0, stratify=strat_data)
         
        
    strat_train = y_train if strat else None
    _, x_gen, _, y_gen = train_test_split(
        x_train, y_train, test_size=gen_ratio / train_ratio, random_state=0, stratify=strat_train)   
        
    if verbose:
        print('complete:  ','{:5d} '.format(len(data)))
        print('train data:','{:5d}  {:5.2f} '.format(len(x_train), len(x_train)/ len(data_x)))
        print('test data: ','{:5d}  {:5.2f} '.format(len(x_test),  len(x_test) / len(data_x)))
        print('gen data:  ','{:5d}  {:5.2f} '.format(len(x_gen),   len(x_gen)  / len(data_x)))

    y_train = y_train.reshape(len(y_train), 1)
    ds_train = pd.DataFrame(np.append(x_train, y_train, axis = 1), columns=data.columns)
    
    y_test = y_test.reshape(len(y_test), 1)
    ds_test = pd.DataFrame(np.append(x_test, y_test, axis = 1), columns=data.columns)
    
    y_gen = y_gen.reshape(len(y_gen), 1)
    ds_gen = pd.DataFrame(np.append(x_gen, y_gen, axis = 1), columns=data.columns)
    
    return ds_train, ds_test, ds_gen

def split_data_from_train(data, target, train_ratio, gen_ratio, strat=True):
    # Aufteilen in Train, Test, Generate
    data_x = data.drop(target, axis = 1).to_numpy()
    data_y = data[target].to_numpy()
    
    strat_data = data_y if strat else None
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, train_size=train_ratio, random_state=0, stratify=strat_data)
 
    strat_train = y_train if strat else None
    _, x_gen, _, y_gen = train_test_split(
        x_train, y_train, test_size=gen_ratio, random_state=0, stratify=strat_train)

    print('complete:  ','{:5d} '.format(len(data)))
    print('train data:','{:5d}  {:5.2f} '.format(len(x_train), len(x_train)/ len(data_x)))
    print('test data: ','{:5d}  {:5.2f} '.format(len(x_test),  len(x_test) / len(data_x)))
    print('gen data:  ','{:5d}  {:5.2f} '.format(len(x_gen),   len(x_gen)  / len(data_x)))

    y_train = y_train.reshape(len(y_train), 1)
    ds_train = pd.DataFrame(np.append(x_train, y_train, axis = 1), columns=data.columns)
    
    y_test = y_test.reshape(len(y_test), 1)
    ds_test = pd.DataFrame(np.append(x_test, y_test, axis = 1), columns=data.columns)
    
    y_gen = y_gen.reshape(len(y_gen), 1)
    ds_gen = pd.DataFrame(np.append(x_gen, y_gen, axis = 1), columns=data.columns)
    
    return ds_train, ds_test, ds_gen

def split_data_val(data, target, train_ratio, test_ratio, val_ratio, gen_ratio, strat=True):
    # Aufteilen in Train, Test, Generate
    data_x = data.drop(target, axis = 1).to_numpy()
    data_y = data[target].to_numpy()

    size = train_ratio + test_ratio + val_ratio
    if size != 1.0:
        print('Daten Verhältnisse ergeben nicht 1.0. Bitte Werte kontrollieren!')
        return
    
    strat_data = data_y if strat else None
    x_train, x_test, y_train, y_test = train_test_split(
        data_x, data_y, train_size=train_ratio, random_state=0, stratify=strat_data)
    
    strat_test = y_test if strat else None
    x_val, x_test, y_val, y_test = train_test_split(
        x_test, y_test, test_size=test_ratio/(test_ratio + val_ratio), random_state=0, stratify=strat_test)

    strat_train = y_train if strat else None
    _, x_gen, _, y_gen = train_test_split(
        x_train, y_train, test_size=gen_ratio / train_ratio, random_state=0, stratify=strat_train)

    print('complete:  ','{:5d} '.format(len(data)))
    print('train data:','{:5d}  {:5.2f} '.format(len(x_train), len(x_train)/ len(data_x)))
    print('test data: ','{:5d}  {:5.2f} '.format(len(x_test),  len(x_test) / len(data_x)))
    print('val data:  ','{:5d}  {:5.2f} '.format(len(x_val),   len(x_val)  / len(data_x)))
    print('gen data:  ','{:5d}  {:5.2f} '.format(len(x_gen),   len(x_gen)  / len(data_x)))

    y_train = y_train.reshape(len(y_train), 1)
    ds_train = pd.DataFrame(np.append(x_train, y_train, axis = 1), columns=data.columns)
    
    y_test = y_test.reshape(len(y_test), 1)
    ds_test = pd.DataFrame(np.append(x_test, y_test, axis = 1), columns=data.columns)
    
    y_val = y_val.reshape(len(y_val), 1)
    ds_val = pd.DataFrame(np.append(x_val, y_val, axis = 1), columns=data.columns)
    
    y_gen = y_gen.reshape(len(y_gen), 1)
    ds_gen = pd.DataFrame(np.append(x_gen, y_gen, axis = 1), columns=data.columns)
    
    return ds_train, ds_test, ds_val, ds_gen
    
# ------------------------------------------------------------------------ #
def run_test_class(train_data, test_data, val_data, target, augmentor, cycles=10, 
                   n_samples=1e3, n_labels=2, verbose=0, weighted=False, balance=False, equal=False):
    train_epochs = 10
    max_plots_line = 10
    
    accs = np.zeros((cycles, 2))
    pres = np.zeros((cycles, 2))
    res = np.zeros((cycles, 2))
    f1s = np.zeros((cycles, 2))
    
    # Datensätze
    x_train = train_data.drop(target, axis=1).to_numpy()
    y_train = train_data[target].to_numpy()
    
    x_test = test_data.drop(target, axis=1).to_numpy()
    y_test = test_data[target].to_numpy()
    
    x_val = val_data.drop(target, axis=1).to_numpy()
    y_val = val_data[target].to_numpy()
    
    x_org = x_train
    x_org = np.append(x_org, x_test, axis=0)
    x_org = np.append(x_org, x_val, axis=0)
    
    # Skalieren
    scaler = skl_prep.MinMaxScaler()
    scaler.fit(x_org)
    
    # Skalieren
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)
    
    val_data = (x_val, y_val)
    
    eval_matrix = np.zeros((cycles*2,4))
    
    # Auswertung
    tab_col = ['Modell', 'Accuracy', 'Precision', 'Recall', 'F1']
    tab_row = ['Orginal', 'Artificial']
    tab_col_line = '{:15} | {:10} | {:10} | {:10} | {:10}'
    tab_line = '{:15} | {:10.3f} | {:10.3f} | {:10.3f} | {:10.3f}'

    for i in range(cycles):
        # Synthetische Daten generieren
        augmentor.generate_syn_data(n_samples, balance=balance, equal=equal)
        ds_data_syn = augmentor.get_syn_data(combine=True)

        # Datensätze
        x_syn = ds_data_syn.drop(target, axis=1).to_numpy()
        y_syn = ds_data_syn[target].to_numpy()
        x_syn = scaler.transform(x_syn)
        
        if weighted:
            _, cn_train = np.unique(y_train, return_counts=True)
            _, cn_syn = np.unique(y_syn, return_counts=True)

            weight_train = cn_train / len(y_train)
            weight_syn = cn_syn / len(y_syn)

            class_weight_train = {0:weight_train[1], 1:weight_train[0]}
            class_weight_syn = {0:weight_syn[1], 1:weight_syn[0]}
        else:
            class_weight_train = None
            class_weight_syn = None
        
        model = create_custom_class_model(x_train.shape[1], n_labels)
        model.fit(x_train, y_train, validation_data=val_data, epochs=train_epochs, 
                  verbose=verbose, class_weight=class_weight_train)
        
        model_syn = create_custom_class_model(x_syn.shape[1], n_labels)
        model_syn.fit(x_syn, y_syn, validation_data=val_data, epochs=train_epochs, 
                      verbose=verbose, class_weight=class_weight_syn)

        # Modell Evaluation
        pred = np.argmax(model.predict(x_test), axis = 1)
        eval_matrix[i+i, 0] = skl_m.accuracy_score(y_test, pred)
        syn_pred = np.argmax(model_syn.predict(x_test), axis = 1)
        eval_matrix[i+i+1, 0] = skl_m.accuracy_score(y_test, syn_pred)
        
        if n_labels > 2: avg = 'micro'
        else: avg = 'binary'
        
        org_loss, org_acc = model.evaluate(x_test, y_test)
        org_pre = skl_m.precision_score(y_test, pred, zero_division=0, average=avg)
        org_rec = skl_m.recall_score(y_test, pred, zero_division=0, average=avg)
        org_f1  = skl_m.f1_score(y_test, pred, zero_division=0, average=avg)
        
        syn_loss, syn_acc = model_syn.evaluate(x_test, y_test)
        syn_pre = skl_m.precision_score(y_test, syn_pred, zero_division=0, average=avg)
        syn_rec = skl_m.recall_score(y_test, syn_pred, zero_division=0, average=avg)
        syn_f1  = skl_m.f1_score(y_test, syn_pred, zero_division=0, average=avg)
       
        accs[i,0] = org_acc
        pres[i,0] = org_pre
        res[i,0]  = org_rec
        f1s[i,0]  = org_f1 
        
        accs[i,1] = syn_acc
        pres[i,1] = syn_pre
        res[i,1]  = syn_rec
        f1s[i,1]  = syn_f1 
        
    print('='* 70)
    print('Evaluation')
    print('='* 70)
    print(tab_col_line.format(*tab_col))
    print('-'* 70)
    for i in range(cycles):
        print(tab_line.format(tab_row[i%1], accs[i,0], pres[i,0], res[i,0], f1s[i,0]))
        print(tab_line.format(tab_row[i%1-1], accs[i,1], pres[i,1], res[i,1], f1s[i,1]))

    print('='* 70)
    print('Org. Accuracy:  %.3f' % (np.average(accs[:,0])))
    print('Org. Precision: %.3f' % (np.average(pres[:,0])))
    print('Org. Recall:    %.3f' % (np.average(res [:,0])))
    print('Org. F1:        %.3f' % (np.average(f1s [:,0])))
    print('Syn. Accuracy:  %.3f' % (np.average(accs[:,1])))
    print('Syn. Precision: %.3f' % (np.average(pres[:,1])))
    print('Syn. Recall:    %.3f' % (np.average(res [:,1])))
    print('Syn. F1:        %.3f' % (np.average(f1s [:,1])))
    
# ------------------------------------------------------------------------ #
def run_test_reg(train_data, test_data, val_data, target, augmentor, cycles=10, n_samples=1e3, verbose=0):
    train_epochs = 10
    max_plots_line = 10
    
    gen_r2 = []
    gen_mse = []
    org_r2 = []
    org_mse = []
    
    # Datensätze
    x_train = train_data.drop(target, axis=1).to_numpy()
    y_train = train_data[target].to_numpy()
    
    x_test = test_data.drop(target, axis=1).to_numpy()
    y_test = test_data[target].to_numpy()
    
    x_val = val_data.drop(target, axis=1).to_numpy()
    y_val = val_data[target].to_numpy()
    
    x_org = x_train
    x_org = np.append(x_org, x_test, axis=0)
    x_org = np.append(x_org, x_val, axis=0)
    
    # Skalieren
    scaler = skl_prep.MinMaxScaler()
    scaler.fit(x_org)
    
    # Skalieren
    x_train = scaler.transform(x_train)
    x_test = scaler.transform(x_test)
    x_val = scaler.transform(x_val)
    
    val_data = (x_val, y_val)
    
    eval_matrix = np.zeros((cycles*2,4))
    
    for i in range(cycles):
        # Synthetische Daten generieren
        augmentor.generate_syn_data(n_samples, classification=False)
        ds_data_syn = augmentor.get_syn_data(combine=True)

        # Datensätze
        x_syn = ds_data_syn.drop(target, axis=1).to_numpy()
        y_syn = ds_data_syn[target].to_numpy()
        x_syn = scaler.transform(x_syn)
            
        model = create_custom_reg_model(x_train, n=3, nodes=8)
        model.fit(x_train, y_train, validation_data=val_data, epochs=train_epochs, verbose=verbose)
        
        model_syn = create_custom_reg_model(x_syn, n=3, nodes=8)
        model_syn.fit(x_syn, y_syn, validation_data=val_data, epochs=train_epochs, verbose=verbose)

        # Modell Evaluation
        org_eval = model.evaluate(x_test, y_test)
        syn_eval = model_syn.evaluate(x_test, y_test)
        
        pred = model.predict(x_test)
        r2 = skl_m.r2_score(y_test, pred)
        mse = skl_m.mean_squared_error(y_test, pred)
        syn_pred = model_syn.predict(x_test)
        r2_syn = skl_m.r2_score(y_test, syn_pred)
        mse_syn = skl_m.mean_squared_error(y_test, syn_pred)
        
        org_r2 = np.append(org_r2, r2)
        gen_r2 = np.append(gen_r2, r2_syn)
        
        org_mse = np.append(org_mse, mse)
        gen_mse = np.append(gen_mse, mse_syn)
        
    print('Org. R2: %.3f (%.3f)' % (np.mean(org_r2), np.std(org_r2)))
    print('Org. MSE: %.3f (%.3f)' % (np.mean(org_mse), np.std(org_mse)))
    print('Syn. R2: %.3f (%.3f)' % (np.mean(gen_r2), np.std(gen_r2)))
    print('Syn. MSE: %.3f (%.3f)' % (np.mean(gen_mse), np.std(gen_mse)))
   
# ------------------------------------------------------------------------ #
def run_class_test_nn(seed_data, target, augmentor, k_model, n_samples, cycles=10, verbose=0):
    train_epochs = 10
    max_plots_line = 10
    train_size = 0.7
    
    accs = np.zeros((cycles, 2))
    pres = np.zeros((cycles, 2))
    res = np.zeros((cycles, 2))
    f1s = np.zeros((cycles, 2))
    
    seed_x = seed_data.drop(target, axis = 1).to_numpy()
    seed_y = seed_data[target].to_numpy()  
    n_classes = len(np.unique(seed_y))
    
    # Aufteilen in Trainings und Test Daten
    seed_x_train, seed_x_test, seed_y_train, seed_y_test = train_test_split(
    seed_x, seed_y, train_size=train_size, random_state=42, stratify=seed_y)

    # Skalieren
    scaler = skl_prep.MinMaxScaler()
    scaler.fit(seed_x)
    
    # Skalieren
    seed_x_train = scaler.transform(seed_x_train)
    seed_x_test = scaler.transform(seed_x_test)
    
    eval_matrix = np.zeros((cycles*2,4))
    
    # Auswertung
    tab_col = ['Modell', 'Accuracy', 'Precision', 'Recall', 'F1']
    tab_row = ['Orginal', 'Artificial']
    tab_col_line = '{:15} | {:10} | {:10} | {:10} | {:10}'
    tab_line = '{:15} | {:10.3f} | {:10.3f} | {:10.3f} | {:10.3f}'

    for i in trange(cycles, desc='running cycle', leave=False):
        # Synthetische Daten generieren
        augmentor.generate_syn_data(n_samples, balance=True)
        ds_data_syn = augmentor.get_syn_data(combine=True)

        # Datensätze
        x_syn = ds_data_syn.drop(target, axis=1).to_numpy()
        y_syn = ds_data_syn[target].to_numpy()
        x_syn = scaler.transform(x_syn)
        
        # model = tfk_model.clone_model(k_model)
        # syn_model = tfk_model.clone_model(k_model)
        
        model = d_test.create_custom_class_model(seed_x_train.shape[1], n_classes, 3, 8)
        model.fit(seed_x_train, seed_y_train, epochs=train_epochs, verbose=verbose)
        
        model_syn = d_test.create_custom_class_model(x_syn.shape[1], n_classes, 3, 8)
        model_syn.fit(x_syn, y_syn, epochs=train_epochs, verbose=verbose)

        # Modell Evaluation
        pred = np.argmax(model.predict(seed_x_test), axis = 1)
        eval_matrix[i+i, 0] = skl_m.accuracy_score(seed_y_test, pred)
        
        syn_pred = np.argmax(model_syn.predict(seed_x_test), axis = 1)
        eval_matrix[i+i+1, 0] = skl_m.accuracy_score(seed_y_test, syn_pred)
        
        if n_classes > 2: avg = 'micro'
        else: avg = 'binary'
        
        org_loss, org_acc = model.evaluate(seed_x_test, seed_y_test)
        org_pre = skl_m.precision_score(seed_y_test, pred, zero_division=0, average=avg)
        org_rec = skl_m.recall_score(seed_y_test, pred, zero_division=0, average=avg)
        org_f1  = skl_m.f1_score(seed_y_test, pred, zero_division=0, average=avg)
        
        syn_loss, syn_acc = model_syn.evaluate(seed_x_test, seed_y_test)
        syn_pre = skl_m.precision_score(seed_y_test, syn_pred, zero_division=0, average=avg)
        syn_rec = skl_m.recall_score(seed_y_test, syn_pred, zero_division=0, average=avg)
        syn_f1  = skl_m.f1_score(seed_y_test, syn_pred, zero_division=0, average=avg)
       
        accs[i,0] = org_acc
        pres[i,0] = org_pre
        res[i,0]  = org_rec
        f1s[i,0]  = org_f1 
        
        accs[i,1] = syn_acc
        pres[i,1] = syn_pre
        res[i,1]  = syn_rec
        f1s[i,1]  = syn_f1 
        
    print('='* 70)
    print('Evaluation')
    print('='* 70)
    print(tab_col_line.format(*tab_col))
    print('-'* 70)
    for i in range(cycles):
        print('Cycle', i)
        print(tab_line.format(tab_row[i%1], accs[i,0], pres[i,0], res[i,0], f1s[i,0]))
        print(tab_line.format(tab_row[i%1-1], accs[i,1], pres[i,1], res[i,1], f1s[i,1]))
    print('='* 70)
    print('Average over', cycles, 'Cycles')
    print('-'* 70)
    print('Org. Accuracy:  %.3f' % (np.average(accs[:,0])), '|', 'Syn. Accuracy:  %.3f' % (np.average(accs[:,1])))
    print('Org. Precision: %.3f' % (np.average(pres[:,0])), '|', 'Syn. Precision: %.3f' % (np.average(pres[:,1])))
    print('Org. Recall:    %.3f' % (np.average(res [:,0])), '|', 'Syn. Recall:    %.3f' % (np.average(res [:,1])))
    print('Org. F1:        %.3f' % (np.average(f1s [:,0])), '|', 'Syn. F1:        %.3f' % (np.average(f1s [:,1])))
    print('='* 70)
   
# ------------------------------------------------------------------------ #
# Funktion für einheitliches ML-Modell
# ------------------------------------------------------------------------ #
def create_custom_class_model(input_dim, output_dim, n=1, nodes=3):
    model = keras.Sequential()
    
    model.add(Input(shape=(input_dim,)))
    for i in range(0, n):
        model.add(Dense(nodes, activation="relu"))
    model.add(Dense(output_dim, activation="softmax"))
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

def create_custom_reg_model(x_data, n=1, nodes=3):
    model = keras.Sequential()

    for i in range(0, n):
        model.add(Dense(nodes, activation="relu"))
    model.add(Dense(1, activation='linear'))
    
    model.compile(optimizer="adam", loss="mse", metrics=["mse"])
    return model