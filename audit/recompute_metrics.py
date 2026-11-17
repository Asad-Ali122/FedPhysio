"""
Independent audit of every FedPhysio number.

Recomputes, from the saved per-window predictions only, the pooled and per-subject metrics of
  * strict LOSO (validation-selected sensor configuration per fold),
  * strict LOSO + causal 30-s smoothing,
  * the subject-calibrated protocol (60 s / 120 s) and strict LOSO on the same windows,
  * the good / average / worst scenarios,
  * every fixed sensor configuration,
and compares them with the published tables in results/tables/.

It does not import the training code; the metric definitions are re-implemented here.

Usage:
    python audit/recompute_metrics.py --runs results/predictions --tables results/tables
"""
import argparse, json, pickle, re, sys
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.metrics import (confusion_matrix, cohen_kappa_score, matthews_corrcoef,
                             roc_auc_score, precision_recall_fscore_support)

K, LAB, RV, SEED, SMOOTH_S = 3, [0, 1, 2], 2, 42, 30

def metrics(y, P):
    yhat = P.argmax(1); cm = confusion_matrix(y, yhat, labels=LAB)
    p, r, f, s = precision_recall_fscore_support(y, yhat, labels=LAB, zero_division=0)
    pres = s > 0
    out = dict(n=int(cm.sum()), acc=np.trace(cm) / cm.sum(), bal_acc=r[pres].mean(), f1=f[pres].mean(),
               kappa=cohen_kappa_score(y, yhat, labels=LAB), mcc=matthews_corrcoef(y, yhat), cm=cm, rec_k=r, sup_k=s)
    out['auc_m'] = roc_auc_score(y, P, multi_class='ovr', average='macro', labels=LAB) if pres.all() else np.nan
    return out

def smooth_causal(P, t):
    order = np.argsort(t); tt = t[order]; PP = P[order]; Ps = np.empty_like(P)
    cs = np.vstack([np.zeros((1, K)), np.cumsum(PP, 0)])
    lo = np.searchsorted(tt, tt - SMOOTH_S + 1e-9, side='left')
    Ps[order] = (cs[np.arange(len(tt)) + 1] - cs[lo]) / (np.arange(len(tt)) + 1 - lo)[:, None]
    return Ps

def load(runs, name, sid):
    d = pd.read_csv(runs / f'{name}_r{RV}_seed{SEED}' / f'S{sid}.csv')
    return d.y_true.values, d[['p0', 'p1', 'p2']].values, d.t_start.values

def pooled(parts):
    subj = {s: metrics(y, P) for s, (y, P) in parts.items()}
    Y = np.concatenate([y for y, _ in parts.values()]); PP = np.concatenate([P for _, P in parts.values()])
    pl = metrics(Y, PP)
    assert (sum(m['cm'] for m in subj.values()) == pl['cm']).all(), 'per-subject matrices do not sum to the pooled matrix'
    assert abs((pl['sup_k'] * pl['rec_k']).sum() / pl['sup_k'].sum() - pl['acc']) < 1e-12, 'accuracy != support-weighted recall'
    return subj, pl

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--runs', default='results/predictions'); ap.add_argument('--tables', default='results/tables')
    a = ap.parse_args(); runs, tables = Path(a.runs), Path(a.tables)
    sel_dir = runs / f'selection_r{RV}_seed{SEED}'
    SEL = {int(f.stem[1:]): json.load(open(f))['selected'] for f in sorted(sel_dir.glob('S*.json'))}
    folds = sorted(SEL); print('folds:', folds); print('selected configuration per fold:', SEL)
    cfg = lambda key, sc='good': f'sub_{key}' + ('' if sc == 'good' else f'__{sc}')
    res, ok = {}, True
    strict = {s: load(runs, cfg(SEL[s]), s) for s in folds}
    res['Strict LOSO (window-level)'] = pooled({s: (y, P) for s, (y, P, t) in strict.items()})
    res['Strict LOSO + causal 30-s smoothing'] = pooled({s: (y, smooth_causal(P, t)) for s, (y, P, t) in strict.items()})
    for cs in [60, 120]:
        fn = runs / f'calibrated_{cs}s_r{RV}_seed{SEED}.pkl'
        if fn.exists():
            C = pickle.load(open(fn, 'rb'))
            res[f'Strict LOSO, same windows as calibrated-{cs}s'] = pooled({s: (C[s][0], C[s][2]) for s in folds})
            res[f'Subject-calibrated ({cs} s per condition)'] = pooled({s: (C[s][0], C[s][1]) for s in folds})
    # compare with the published protocol table
    pub = pd.read_csv(tables / 'v2_protocols.csv').set_index('protocol')
    print('\n%-46s %6s %8s %8s %8s %8s   %s' % ('protocol', 'n', 'acc', 'f1', 'kappa', 'auc', 'matches table'))
    for k, (subj, pl) in res.items():
        match = k in pub.index and all(abs(pl[m] - pub.loc[k, m]) < 5e-5 for m in ['acc', 'f1', 'kappa', 'mcc', 'bal_acc', 'auc_m'])
        ok &= match
        print('%-46s %6d %8.4f %8.4f %8.4f %8.4f   %s' % (k, pl['n'], pl['acc'], pl['f1'], pl['kappa'], pl['auc_m'], 'YES' if match else 'NO'))
    subj, pl = res['Strict LOSO (window-level)']
    print('\npooled integer confusion matrix (rows true B/S/A, columns predicted):\n', pl['cm'])
    print('support-weighted recall = %.6f, accuracy = %.6f' % ((pl['sup_k'] * pl['rec_k']).sum() / pl['sup_k'].sum(), pl['acc']))
    # scenarios
    pubs = pd.read_csv(tables / 'v2_scenarios.csv').set_index('scenario')
    print('\nscenario   acc      f1       kappa    matches table')
    for sc in ['good', 'average', 'worst']:
        _, p2 = pooled({s: load(runs, cfg(SEL[s], sc), s)[:2] for s in folds})
        m = all(abs(p2[x] - pubs.loc[sc, x]) < 5e-5 for x in ['acc', 'f1', 'kappa']); ok &= m
        print('%-9s %.4f   %.4f   %.4f   %s' % (sc, p2['acc'], p2['f1'], p2['kappa'], 'YES' if m else 'NO'))
    # fixed sensor configurations
    pubc = pd.read_csv(tables / 'v2_subsets.csv').set_index('subset')
    print('\nconfiguration                        acc      kappa    matches table')
    for key in pubc.index:
        _, p3 = pooled({s: load(runs, cfg(key), s)[:2] for s in folds})
        m = all(abs(p3[x] - pubc.loc[key, x]) < 5e-5 for x in ['acc', 'f1', 'kappa']); ok &= m
        print('%-36s %.4f   %.4f   %s' % (key, p3['acc'], p3['kappa'], 'YES' if m else 'NO'))
    print('\nALL PUBLISHED NUMBERS REPRODUCED' if ok else '\nMISMATCH FOUND -- see rows marked NO')
    sys.exit(0 if ok else 1)

if __name__ == '__main__':
    main()
