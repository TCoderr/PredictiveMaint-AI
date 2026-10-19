"""PredictiveMaint AI tests: independent recomputation of the notebook's results. Run: python -m pytest -v"""
import json
import zipfile
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, fbeta_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
R = json.loads((ROOT / 'reports' / 'metrics.json').read_text(encoding='utf-8'))
TARGET, MODES = 'Machine failure', ['TWF', 'HDF', 'PWF', 'OSF', 'RNF']
NUM = ['Air temperature [K]', 'Process temperature [K]', 'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
LEAKY = {'UDI', 'Product ID', TARGET, *MODES}


@pytest.fixture(scope='module')
def df():
    with zipfile.ZipFile(ROOT / 'data' / 'raw' / 'ai4i2020.zip') as z:
        d = pd.read_csv(z.open('ai4i2020.csv'), encoding='utf-8-sig')
    d['temp_diff'] = d['Process temperature [K]'] - d['Air temperature [K]']
    d['power_w'] = d['Torque [Nm]'] * d['Rotational speed [rpm]'] * 2 * np.pi / 60
    d['strain'] = d['Tool wear [min]'] * d['Torque [Nm]']
    return d


@pytest.fixture(scope='module')
def splits(df):
    cols = R['tuning']['Random Forest']['features']
    X, y = df[cols], df[TARGET]
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test


def pre(cols):
    return ColumnTransformer([('num', StandardScaler(), [c for c in cols if c != 'Type']),
                              ('cat', OneHotEncoder(handle_unknown='ignore'), ['Type'])])


def opt(v):
    return None if v == 'None' else (int(v) if v.isdigit() else v)


# ------------------------------------------------------------ data facts
def test_dataset_facts(df):
    assert len(df) == R['data']['rows'] == 10000
    assert int(df[NUM + ['Type', TARGET]].isnull().sum().sum()) == 0 == R['data']['missing']
    assert int(df[TARGET].sum()) == R['data']['failures'] == 339
    assert {m: int(df[m].sum()) for m in MODES} == R['data']['mode_counts']


def test_label_inconsistencies_are_reported_correctly(df):
    any_mode = df[MODES].max(axis=1)
    assert int(((df[TARGET] == 1) & (any_mode == 0)).sum()) == R['data']['failure_without_mode']
    assert int(((df[TARGET] == 0) & (any_mode == 1)).sum()) == R['data']['mode_without_failure']


# ------------------------------------------------------------ leakage
def test_no_leaky_columns_in_any_model():
    for name, t in R['tuning'].items():
        assert not LEAKY & set(t['features']), name


def test_stratified_disjoint_split(splits):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    assert (len(X_train), len(X_val), len(X_test)) == (6000, 2000, 2000)
    assert not (set(X_train.index) & set(X_val.index) or set(X_train.index) & set(X_test.index) or set(X_val.index) & set(X_test.index))
    rates = [y_train.mean(), y_val.mean(), y_test.mean()]
    assert max(rates) - min(rates) < 0.002                     # stratification keeps the 3.4 % failure rate
    assert [R['split'][k]['failures'] for k in ('train', 'val', 'test')] == [int(y_train.sum()), int(y_val.sum()), int(y_test.sum())]


def test_scaler_is_fit_on_training_data_only(splits):
    X_train, X_val, *_ = splits
    cols = R['tuning']['Random Forest']['features']
    p = pre(cols).fit(X_train[cols])
    num = [c for c in cols if c != 'Type']
    assert np.allclose(p.named_transformers_['num'].mean_, X_train[num].mean().values)
    assert not np.allclose(p.named_transformers_['num'].mean_, pd.concat([X_train, X_val])[num].mean().values)


# ------------------------------------------------------------ independent recomputation of reported numbers
def test_logreg_baseline_reproduces(df, splits):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    t = R['tuning']['LogReg (temel)']; cols = t['features']
    Xtr, Xva, Xte = (df.loc[a.index, cols] for a in (X_train, X_val, X_test))
    m = Pipeline([('pre', pre(cols)), ('model', LogisticRegression(max_iter=3000, C=float(t['best_params']['C']),
                                                                   class_weight=opt(t['best_params']['class_weight'])))]).fit(Xtr, y_train)
    assert average_precision_score(y_val, m.predict_proba(Xva)[:, 1]) == pytest.approx(t['val_pr_auc'], abs=1e-9)
    pred = m.predict(Xte)
    assert recall_score(y_test, pred) == pytest.approx(R['test_default']['LogReg (temel)']['recall'], abs=1e-9)


def test_random_forest_and_threshold_reproduce(splits):
    X_train, X_val, X_test, y_train, y_val, y_test = splits
    t = R['tuning']['Random Forest']; bp = t['best_params']; cols = t['features']
    rf = Pipeline([('pre', pre(cols)), ('model', RandomForestClassifier(
        n_estimators=300, max_depth=opt(bp['max_depth']), min_samples_leaf=int(bp['min_samples_leaf']),
        class_weight=opt(bp['class_weight']), n_jobs=-1, random_state=42))]).fit(X_train, y_train)
    p_val, p_test = rf.predict_proba(X_val)[:, 1], rf.predict_proba(X_test)[:, 1]
    assert average_precision_score(y_val, p_val) == pytest.approx(t['val_pr_auc'], abs=1e-9)
    # threshold must be the F2-maximiser on VALIDATION data (test data never used for the choice)
    thrs = np.round(np.arange(0.05, 0.951, 0.01), 2)
    f2 = [fbeta_score(y_val, (p_val >= th).astype(int), beta=2) for th in thrs]
    assert float(thrs[int(np.argmax(f2))]) == R['threshold']['value']
    pred = (p_test >= R['threshold']['value']).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    fin = R['test_final']
    assert (tn, fp, fn, tp) == (fin['tn'], fin['fp'], fin['fn'], fin['tp'])
    assert precision_score(y_test, pred) == pytest.approx(fin['precision'], abs=1e-9)
    assert fin['fnr'] == pytest.approx(1 - fin['recall'], abs=1e-12)        # FNR = 1 - recall


def test_selected_model_is_best_on_validation_not_test():
    val = {n: t['val_pr_auc'] for n, t in R['tuning'].items()}
    assert R['selected_model'] == max(val, key=val.get)


def test_accuracy_paradox_documented():
    naive = R['naive_all_zero']
    assert naive['accuracy'] > 0.96 and naive['recall'] == 0 and naive['fn'] == R['split']['test']['failures']


# ------------------------------------------------------------ notebook + figures
def test_notebook_executed_without_errors():
    nb = nbformat.read(ROOT / 'notebooks' / 'PredictiveMaint_AI.ipynb', as_version=4)
    code = [c for c in nb.cells if c.cell_type == 'code']
    assert code and all(c.execution_count for c in code)
    assert not [o for c in code for o in c.get('outputs', []) if o.output_type == 'error']


def test_at_least_three_eda_figures():
    figs = {p.name for p in (ROOT / 'reports' / 'figures').glob('*.png')}
    assert {'01_sinif_dagilimi.png', '02_ozellik_dagilimlari.png', '03_arizayla_iliskili.png'} <= figs
    assert {'05_karisiklik_matrisleri.png', '08_final_karisiklik.png'} <= figs
