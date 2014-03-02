#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run an experiment predicting county statistics from tweets.
"""
import argparse
import copy
import csv
import io
from itertools import groupby
import logging
import os
import pickle
import random
import re
import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import f_regression
import utm

import data
import spatial

logger = logging.getLogger(__name__)


def read_targets(filename, counties):
    """ Read target health outcomes from file. """
    targets = dict()
    reader = csv.DictReader(open(filename), delimiter='\t')
    target_alpha = [v for v in reader.fieldnames if v != 'FIPS']
    for line in reader:
        fips = line['FIPS']
        del line['FIPS']
        if fips in counties:
            targets[fips] = [float(line[k]) for k in target_alpha]
    return targets, target_alpha


def read_dict(f):
    """ Read a file to a dict. """
    d = {}
    with io.open(f, mode='r', encoding='utf8') as fp:
        for line in fp:
            (v, k) = line.split()
            d[k] = v
    return d


def read_states(filename, counties):
    return read_dict(filename)


def to_matrices(counties, targets):
    """
    >>> X, Y = to_matrices({'cty1': [1, 2], 'cty2': [3, 4]}, {'cty1': [5, 6], 'cty2': [7, 8]})
    >>> print X
    [[1 2]
     [3 4]]
    >>> print Y
    [[5 6]
     [7 8]]
    """
    X = []
    Y = []
    for cty in sorted(counties):
        X.append(np.array(counties[cty]))
        Y.append(np.array(targets[cty]))
    return np.array(X), np.array(Y)


def linreg(x, y, train, test):
    m = linear_model.LinearRegression(fit_intercept=True)
    m.fit(x[train], y[train])
    return (m, m.predict(x[train]), m.predict(x[test]))


def ridge(x, y, train, test):
    m = linear_model.Ridge(.1)
    m.fit(x[train], y[train])
    return (m, m.predict(x[train]), m.predict(x[test]))


def lasso(x, y, train, test):
    m = linear_model.Lasso(alpha=0.01)
    m.fit(x[train], y[train])
    return (m, m.predict(x[train]), m.predict(x[test]))


def mse(truth, pred):
    return np.mean((pred - truth) * (pred - truth))


def feature_correls(X, y, coords):
    """ Compute correlations between each feature and the label, correcting
    for spatial auto-correlaiton. """
    fvals, pvals = f_regression(X, y)
    for i in range(len(X[0])):
        r, p = spatial.spatial_correlation(X[:, i], y, coords)
        pvals[i] = p
    return fvals, pvals


def run_expt(model_f, counties, states, targets, data_alpha, target_alpha, args, coords):
    cv = make_state_cv(counties, states, 5)
    X, Y = to_matrices(counties, targets)
    county_ids = np.array(sorted(counties))
    for yi, ylabel in enumerate(target_alpha):
        y = Y[:, yi]
        train_folds = []
        test_folds = []
        for train, test in cv:
            m, train_pred, test_pred = model_f(X, y, train, test)
            train_folds.append((y[train], train_pred, county_ids[train]))
            test_folds.append((y[test], test_pred, county_ids[test]))
        m, train_pred, test_pred = model_f(X, y, range(len(y)), range(len(y)))
        # fvals, pvals = f_regression(X, y)
        fvals, pvals = feature_correls(X, y, [coords[cty] for cty in county_ids])
        write_results(train_folds, test_folds, m, fvals, pvals, data_alpha, ylabel, args, coords)
    write_aggregate_results(args)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def write_tsv(results, filen):
    fp = io.open(filen, mode='wt')
    fp.write(u'\t'.join(str(k) for k in sorted(results.keys())) + '\n')
    fp.write(u'\t'.join(str(results[k]) for k in sorted(results.keys())) +
             '\n')
    fp.close()


def plot_scatter(prefix, y, pred):
    fit = np.polyfit(y, pred, 1)
    fit_fn = np.poly1d(fit)
    plt.plot(y, pred, 'yo', y, fit_fn(y), '--k')
    plt.xlabel('truth')
    plt.ylabel('predicted')
    plt.savefig(prefix + '/scatter.pdf')
    plt.close()


def print_words2(vocab, coef, fvals, pvals, n, fp):
    top_indices = coef.argsort()[-n:][::-1]
    for i in top_indices:
        fp.write('%s=%f\n' % (vocab[i], coef[i]))
    bot_indices = coef.argsort()[0:n:][::-1]
    for i in bot_indices:
        fp.write('%s=%f\n' % (vocab[i], coef[i]))
    fp.close()


def print_words(vocab, coef, fvals, pvals, n, fp):
    top_indices = pvals.argsort()
    for i in top_indices:
        fp.write('%s\t%e\t%e\t%e\n' % (vocab[i], coef[i], fvals[i], pvals[i]))
    fp.close()


all_results = []


def fmt_target(s):
    return re.sub(r'\s+', '', re.sub(r'[^\w\s]+', '',
                                     s.strip())).lower()


def write_results(train_folds, test_folds, m, fvals, pvals, vocab, target_name, args, coords):
    """ Write prediction results to disk. """
    target_name = fmt_target(target_name)
    prefix = args.input + '.out/' + target_name
    mkdir(prefix)
    pickle.dump(args, open(prefix + '/args.pkl', 'wb'))
    pickle.dump(train_folds, open(prefix + '/train_folds.pkl', "wb"))
    pickle.dump(test_folds, open(prefix + '/test_folds.pkl', "wb"))
    train_results = evaluate(train_folds, coords)
    test_results = evaluate(test_folds, coords)
    logger.info('%20s:\ttrain\t%s' % (target_name[:20], fmt_eval(train_results, ['smape_m', 'correl', 'correl_a'])))
    logger.info('%20s:\ttest\t%s' % (target_name[:20], fmt_eval(test_results, ['smape_m', 'correl', 'correl_a'])))

    global all_results
    all_results.append(test_results)
    pickle.dump(test_results, open(prefix + '/test_results.pkl', 'wb'))
    pickle.dump(train_results, open(prefix + '/train_results.pkl', 'wb'))
    write_tsv(test_results, prefix + '/test_results.tsv')
    write_tsv(train_results, prefix + '/train_results.tsv')

    plot_scatter(prefix, [item for t in test_folds for item in t[0]],
                 [item for t in test_folds for item in t[1]])

    fp = io.open(prefix + '/words.txt', mode='wt', encoding='utf8')
    if len(m.coef_.shape) > 1:
        for ci, c in enumerate(m.coef_):
            print_words(vocab, c, fvals, pvals, 100, fp)
    else:
        print_words(vocab, m.coef_, fvals, pvals, 200, fp)


def avg_results():
    """ Average all_results. """
    global all_results
    # FIXME: remove global nonesense.
    result = {}
    for key in all_results[0].keys():
        result[key] = np.mean([d[key] for d in all_results])
    return result


def write_aggregate_results(args):
    """ Write results aggregated over health outcomes. """
    aggregate = avg_results()
    pickle.dump(args, open(args.input + '.out/args.pkl', 'wb'))
    pickle.dump(aggregate, open(args.input + '.out/results.pkl', 'wb'))
    write_tsv(aggregate, args.input + '.out/results.tsv')


def fmt_eval(results, keys):
    """ Format evaluation output. """
    return '\t'.join('%s=%.4f' % (k, results[k]) for k in keys)


def smape(tr, pr):
    '''Symmetric mean absolute percentage error. Returns number in [0,1].  See
    the third version of the formula here:
    http://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    >>> smape([100, 200], [110, 100]) # doctest:+ELLIPSIS
    0.215...
    '''
    return 1.0 * sum([abs(p - t) for (t, p) in zip(tr, pr)]) / \
        (sum(tr) + sum(pr))


def evaluate(folds, coords):
    """ Each element of folds is (truth, predicted, county_id). """
    results = dict()
    mse = [np.mean((t[0] - t[1]) * (t[0] - t[1])) for t in folds]
    smapes = [smape(t[0], t[1]) for t in folds]
    results['mse_m'] = np.mean(mse)
    results['mse_std'] = np.std(mse)
    results['smape_m'] = np.mean(smapes)
    results['smape_std'] = np.std(smapes)
    correl = spatial.spatial_correlation([item for t in folds for item in t[0]],
                                         [item for t in folds for item in t[1]],
                                         [coords[item] for t in folds for item in t[2]])
    results['correl'] = correl[0]
    results['correl_a'] = correl[1]
    return results


def make_state_cv(counties, states, k=5, seed=12345):
    """ Make cross validation folds where counties are grouped by state.
    >>> cv = make_state_cv({'ct1': 0, 'ct2': 0, 'ct3': 0, 'ct4': 0, 'ct5': 0, 'ct6': 0}, {'ct1': 's1', 'ct2': 's2', 'ct3': 's1', 'ct4': 's2', 'ct5': 's3', 'ct6': 's3'}, 3, 12345)
    >>> sorted(cv[0][0])
    [0, 2, 4, 5]
    >>> sorted(cv[0][1])
    [1, 3]
    >>> sorted(cv[1][0])
    [1, 3, 4, 5]
    >>> sorted(cv[1][1])
    [0, 2]
    >>> sorted(cv[2][0])
    [0, 1, 2, 3]
    >>> sorted(cv[2][1])
    [4, 5]
    """
    random.seed(seed)
    county_ids = sorted(counties)
    groups = [list(group[1]) for group in groupby(sorted(county_ids, key=lambda c: states[c]), key=lambda c: states[c])]
    random.shuffle(groups)
    groups = [[county_ids.index(c) for c in g] for g in groups]
    folds = []
    test_n = len(counties) / k
    canadd = copy.copy(groups)
    while len(canadd) > 0:
        test_set = canadd.pop()
        while len(test_set) < test_n and len(canadd) > 0:
            test_set.extend(canadd.pop())
        train_set = [t for t in range(len(county_ids)) if t not in test_set]
        folds.append([train_set, test_set])
    return folds


def add_controls(controls, counties, alpha, targets, target_alpha, controls_only):
    """ Add control variables (race, etc) to data matrix. """
    control_names = [unicode(c) for c in controls.split(',')]
    control_indices = [target_alpha.index(n) for n in control_names]
    if controls_only:  # delete all other features
        del alpha[:]
        for c in control_names:
            alpha.append(c)
        for c in counties:
            counties[c] = np.array(targets[c])[control_indices]
    else:  # append controls to other features
        alpha += control_names
        for c in counties:
            counties[c].extend(np.array(targets[c])[control_indices])
    print 'example of last 10 features are:', counties.values()[0][-10:]


def read_coords(filename):
    """ Return a map from county to lat/long. """
    d = {}
    for row in csv.reader(open(filename, 'rb'), delimiter='\t'):
        # d[row[1]] = (float(row[10]), float(row[11]))
        d[row[1]] = utm.from_latlon(float(row[10]), float(row[11]))[:2]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--controls', default=None, help='comma separated list of column names from --targets to use as control variables ...')
    ap.add_argument('--controls-only', action='store_true', default=False, help='restrict features to control variables')
    ap.add_argument('--coords', default='/data/twcounty/stats2/counties/counties.top100.bounding.txt', help='tsv file with counties, lat/long, and bounding boxes')
    ap.add_argument('--input', default='/data/twcounty/counties.pkl', help='pickled county feature Data object')
    ap.add_argument('--model', default='ridge', help='name of function to build model [linreg, ridge, lasso]')
    ap.add_argument('--states', default='/data/twcounty/states.tsv', help='file like <state> <county_id>')
    ap.add_argument('--targets', default='/data/twcounty/targets.tsv', help='targets per county')
    args = ap.parse_args()
    coords = read_coords(args.coords)
    logger.info('args=%s' % str(args))
    logger.info('reading data from %s' % args.input)
    county_data = data.read(args.input)
    counties = county_data.features
    data_alpha = sorted(county_data.alpha.keys(), key=lambda k: county_data.alpha[k])
    logger.info('reading targets from %s' % args.targets)
    targets, target_alpha = read_targets(args.targets, counties.keys())
    logger.info('reading states from %s' % args.states)
    states = read_states(args.states, counties.keys())
    logger.info('targets for %s are %s' % (targets.keys()[0], str(targets[targets.keys()[0]])))
    logger.info('running experiment on %d counties' % len(counties))
    if args.controls:
        add_controls(args.controls, counties, data_alpha, targets, target_alpha, args.controls_only)
    model_f = getattr(sys.modules[__name__], args.model)
    run_expt(model_f, counties, states, targets, data_alpha, target_alpha, args, coords)


if __name__ == '__main__':
    main()
