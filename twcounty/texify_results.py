#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Make figures and TeX tables from stored experimental results. This makes it
easy to regenerate most of the tables and figures for the paper with one
command. However, it needs some serious refactoring to become legible. It also
hard-codes the path to the experimental results to /data/twcounty/features/.
"""
import codecs
from collections import defaultdict
import copy
import math
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import pysal
from scipy.spatial import distance
from scipy.stats import wilcoxon
import statsmodels.api as sm

import data as twdata
import expt as twexpt
import spatial as twspatial

labels = ['ambulatorycaresensitiveconditions', 'childreninpoverty', 'chlamydiarate', 'dentistratio', 'excessivedrinking', 'fairpoorhealth', 'fastfoods', 'hba1c', 'highschoolgradrateafgr', 'limitedaccesstohealthyfoods', 'lowbirthweight', 'mammography', 'mentallyunhealthydays', 'mvmortalityrate', 'nosocialemotionalsupport', 'obese', 'physicallyinactive', 'physicallyunhealthydays', 'primarycarephys', 'recfacilityrate', 'singleparenthouseholds', 'smokers', 'somecollege', 'teenbirthrate', 'unemployed', 'uninsured', 'violentcrimerate']

label_map = {'ambulatorycaresensitiveconditions': 'Ambulatory Care', 'childreninpoverty': 'Child Poverty', 'chlamydiarate': 'Chlamydia', 'dentistratio': 'Dentist Access', 'excessivedrinking': 'Drinking', 'fairpoorhealth': 'Poor Health', 'fastfoods': 'Fast Food', 'hba1c': 'Diabetes', 'highschoolgradrateafgr': 'HS Grad Rate', 'limitedaccesstohealthyfoods': 'Limited Healthy Food', 'lowbirthweight': 'Low Birth Weight', 'mammography': 'Mammography', 'mentallyunhealthydays': 'Mentally Unhealthy', 'mvmortalityrate': 'Vehicle Mortality', 'nosocialemotionalsupport': 'No socio-emotional support', 'obese': 'Obesity', 'physicallyinactive': 'Inactivity', 'physicallyunhealthydays': 'Unhealthy Days', 'primarycarephys': 'Primary Care', 'recfacilityrate': 'Rec Facilities', 'singleparenthouseholds': 'Single Parent', 'smokers': 'Smokers', 'somecollege': 'College', 'teenbirthrate': 'Teen Births', 'unemployed': 'Unemployment', 'uninsured': 'No Insurance', 'violentcrimerate': 'Violent Crime'}

control_labels = ['< 18', '65 and over', 'Female', 'Afro-hispanic', 'med_income']

path = '/data/twcounty/features/'


def avg_results(all_results):
    result = {}
    for key in all_results[0].keys():
        result[key] = np.mean([d[key] for d in all_results])
    return result


def b(s):
    """ bold """
    return '{\\bf ' + s + '}'


def tab_features():
    """ Write summary results for different variable combinations:
    - liwc
    - perma
    - liwc + perma
    - controls
    - controls + liwc + perma
    """
    outf = open('paper/tables/features.tex', 'w')
    dirs = [((path + d), n) for d, n in [('ridge/counties.norm=user.perma.pkl.out', b('PERMA')),
                                         ('ridge/counties.norm=user.liwc.pkl.out', b('LIWC')),
                                         ('ridge/counties.norm=user.liwc.perma.pkl.out', b('LIWC+PERMA')),
                                         ('ridge_control_only/counties.norm=none.unigrams.pkl.out', b('Controls')),
                                         ('ridge_control_and_feats/counties.norm=user.perma.pkl.out', b('Controls+PERMA')),
                                         ('ridge_control_and_feats/counties.norm=user.liwc.pkl.out', b('Controls+LIWC')),
                                         ('ridge_control_and_feats/counties.norm=user.liwc.perma.pkl.out', b('Controls+LIWC+PERMA'))]]
    outf.write('\\begin{table}[t]\n\\centering\n\\begin{tabular}{|r|c|c|}\n\hline\n')
    outf.write('%30s\t& %6s\t& %6s\\\\\n\\hline\n' % (b('Variables'), b('r'), b('SMAPE')))
    for di, name in dirs:
        results = [pickle.load(open(di + '/' + li + '/test_results.pkl', 'rb')) for li in labels]
        results = avg_results(results)
        outf.write('%30s\t& %.2f \t& %.2f\\%%\\\\\n' % (name, results['correl'], results['smape_m'] * 100.))
    outf.write('\\hline\n\\end{tabular}\n')
    outf.write('\\caption{Held-out correlation and SMAPE averaged across all 27 output variables using various combinations of input variables. All models use {\\bf User} normalization. \\label{tab.features}}\n')
    outf.write('\\end{table}\n')
    outf.close()


def tab_norm():
    """ Output summary results for different normalizations of liwc + perma model."""
    outf = open('paper/tables/norm.tex', 'w')
    dirs = [((path + d), n) for d, n in [
            ('ridge_control_and_feats/counties.norm=none.liwc.perma.pkl.out', b('None')),
            ('ridge_control_and_feats/counties.norm=none.liwc.perma.log.pkl.out', b('Log')),
            ('ridge_control_and_feats/counties.norm=word.liwc.perma.pkl.out', b('Word')),
            ('ridge_control_and_feats/counties.norm=user.liwc.perma.pkl.out', b('User'))]]

    outf.write('\\begin{table}[t]\n\\centering\n\\begin{tabular}{|r|c|c|}\n\hline\n')
    outf.write('%30s\t& %6s\t& %6s\\\\\n\\hline\n' % (b('Norm'), b('r'), b('SMAPE')))
    for di, name in dirs:
        results = [pickle.load(open(di + '/' + li + '/test_results.pkl', 'rb')) for li in labels]
        results = avg_results(results)
        outf.write('%30s\t& %.2f \t& %.2f\\%%\\\\\n' % (name, results['correl'], results['smape_m'] * 100.))
    outf.write('\\hline\n\\end{tabular}\n')
    outf.write('\\caption{Held-out correlation and SMAPE for the Controls+LIWC+PERMA model averaged across all 27 output variables using various normalization strategies. \\label{tab.norm}}\n')
    outf.write('\\end{table}\n')
    outf.close()


def pstar(value, bonfer):
    """ significance marker, bonferonni corrected """
    if value <= .001 / bonfer:
        return '***'
    elif value <= .01 / bonfer:
        return '**'
    elif value <= .05 / bonfer:
        return '*'
    elif value <= .1 / bonfer:
        return '$\\circ$'
    else:
        return ''


def ctr(s, pipe='|'):
    """ Make a centered table header. """
    return '\\multicolumn{1}{c%s}{%s}' % (pipe, s)


def test_wilcox_smape(path1, path2):
    folds1 = pickle.load(open(path1, 'rb'))
    folds2 = pickle.load(open(path2, 'rb'))
    smape1 = [twexpt.smape(t[0], t[1]) for t in folds1]
    smape2 = [twexpt.smape(t[0], t[1]) for t in folds2]
    return wilcoxon(smape1, smape2)


def read_control_results():
    return [(label,
             pickle.load(open(path + 'ridge/counties.norm=user.liwc.perma.pkl.out' + '/' + label + '/test_results.pkl', 'rb')),
             pickle.load(open(path + 'ridge_control_only/counties.norm=none.unigrams.pkl.out' + '/' + label + '/test_results.pkl', 'rb')),
             pickle.load(open(path + 'ridge_control_and_feats/counties.norm=user.liwc.perma.pkl.out' + '/' + label + '/test_results.pkl', 'rb')))
            for label in labels]


def tab_outcomes():
    """ Comparison of twitter and control models across each of the
    outcomes. Print big table with correlation and SMAPE. """
    results = read_control_results()
    outf = open('paper/tables/outcomes.tex', 'w')
    outf.write('\\begin{table*}[t]\n\\centering\n\\begin{tabular}{|r|l|l|l|r||r|r|r|r|}\n\hline\n')
    outf.write(' & \\multicolumn{4}{c||}{%s} & \\multicolumn{4}{c|}{%s} \\\\\n \\hline \n' % (b('Pearson\'s r'), b('SMAPE')))
    outf.write('%s &\t %s &\t %s &\t %s &\t %s &\t %s &\t %s &\t %s &\t %s\\\\\n\\hline\n' % (b('Outcome'), ctr(b('T')), ctr(b('C')), ctr(b('T+C')), ctr('$\\Delta$', '||'), ctr(b('T')), ctr(b('C')), ctr(b('T+C')), ctr('$\\Delta$')))
    bonfer = len(labels)  # Bonferroni correction factor.
    for label, noctl, ctl_only, both in sort_by_improvement(results):
        wilcox = test_wilcox_smape(
            path + 'ridge_control_only/counties.norm=none.unigrams.pkl.out/' + label + '/test_folds.pkl',
            path + 'ridge_control_and_feats/counties.norm=user.liwc.perma.pkl.out/' + label + '/test_folds.pkl')
        print label, 'wilcox=', wilcox
        wilcox_sig = pstar(wilcox[1], 1)
        label = label_map[label]
        outf.write('%30s \t & %.2f%s \t & %.2f%s \t & %.2f%s \t & %.0f\\%% \t & %.1f\\%% $\\pm$ %.1f \t & %.1f\\%% $\\pm$ %.1f \t & %.1f\\%% $\\pm$ %.1f \t & %.0f\\%%%s\\\\\n' %
                   (label,
                    noctl['correl'], pstar(noctl['correl_a'], bonfer),
                    ctl_only['correl'], pstar(ctl_only['correl_a'], bonfer),
                    both['correl'], pstar(both['correl_a'], bonfer),
                    (both['correl'] - ctl_only['correl']) / ctl_only['correl'] * 100.,
                    noctl['smape_m'] * 100., noctl['smape_std'] * 100.,
                    ctl_only['smape_m'] * 100., ctl_only['smape_std'] * 100.,
                    both['smape_m'] * 100., both['smape_std'] * 100.,
                    (ctl_only['smape_m'] - both['smape_m']) / ctl_only['smape_m'] * 100.,
                    wilcox_sig))

    outf.write('\\hline\n\\end{tabular}\n')
    outf.write('\\caption{Held-out correlation and mean SMAPE (with standard deviation) for each outcome under three models --- {\\sl T}: Twitter model using LIWC and PERMA lexicons; {\\sl C}: control variables (age, gender, race, income); {\\sl T+C}: Twitter and controls.  All models use {\\bf User} normalization. $\\Delta$ is the percent relative improvement (either correlation or SMAPE) from model {\\sl C} to {\\sl T+C}, an estimate of how complementary the two models are.  Pearson\'s $r$ significance is indicated by $\\circ = 0.1$, $\\ast =  0.05$, $\\ast\\ast = 0.01$, $\\ast\\ast\\ast = 0.001$ (degress of freedom = 98). The thresholds have been Bonferroni-corrected (using the 27 outcomes).\\label{tab.outcomes}}\n')
    outf.write('\\end{table*}\n')
    outf.close()


def z2r(z):
    """ Compute a correlation coefficient r given a Fisher's Z score.
    value. """
    return (math.exp(2. * z) - 1) / (math.exp(2. * z) + 1)


def r2z(r):
    """ Compute Fisher's Z score given a correlation coefficient r. """
    return .5 * math.log((1 + r) / (1 - r))


def r_ci(r, n):
    """ Compute confidence intervals for a given correlation coefficient and
    sample size (alpha fixed at .95). See
    http://www.tc3.edu/instruct/sbrown/stat/correl.htm
    >>> r_ci(.84, 25)  # doctest: +ELLIPSIS
    (0.66..., 0.92...)
    """
    z = r2z(r)
    error = 1. / math.sqrt(n - 3) * 1.96
    return (z2r(z - error), z2r(z + error))


def f2r(f, n):
    """ Convert F-statistic to r (correlation). p=number of
    coefficients, n=sample size

    >>> f2r(18.616645974174119, 100)  # doctest: +ELLIPSIS
    0.399...
    """
    return math.sqrt(f / (n - 2 + f))
    #return math.sqrt(1. - 1. / (1. + f * p / n))


def read_tsv(filename, header):
    """ Read words.tsv file.  Header used to make a dict from each line.
    format: word coef f alpha"""
    fp = codecs.open(filename, 'r', 'utf8')
    lines = []
    for line in fp:
        parts = line.strip().split('\t')
        d = dict([(header[i + 1], float(v)) for i, v in enumerate(parts[1:])])
        d[header[0]] = parts[0]
        lines.append(d)
    return lines


def plot_words(words, label, total_words):
    """ Plot correlations with error bars for each word for one outcome (label)."""
    words = [w for w in reversed(words)]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tick_params(axis='y', labelsize='22', pad=10, left=False)
    ax.tick_params(axis='x', labelsize='16')
    plt.axvline(x=0, color='k', ls=':')  # vertical line
    plt.title(label_map[label], size='22')
    for i, w in enumerate(words):
        plt.errorbar(w['r'], i + 1, xerr=w['r_err'], color='b', elinewidth=2, capsize=5)
        plt.plot(w['r'], i + 1, 'bo', ms=10)
    ax.set_yticklabels([' '] + ['%s' % (w['var'] + ' ' + pstar(w['alpha'], total_words * len(labels))) for w in words])
    plt.yticks(np.arange(len(words) + 2))
    ax.spines['left'].set_visible(False)
    plt.xticks([-.8, -.4, 0, .4, .8])
    plt.savefig('paper/figs/' + label + '.pdf', bbox_inches='tight')
    plt.close()


def write_subfig(label, outf):
    outf.write('\\begin{subfigure}[b]{0.33\\textwidth}\n' +
               '\\centering\n' +
               '\\includegraphics[width=\\columnwidth,height=.6\\columnwidth]{figs/' + label + '}\n' +
               '\\label{fig.' + label + '}\n' +
               '\\end{subfigure}\n')


def sort_by_improvement(results):
    return sorted(results, key=lambda x: (x[2]['correl'] - x[3]['correl']) / x[2]['correl'])


def fig_top_predictors():
    """ Make a figure showing the most highly correlated lexical cateogories
    for each highly correlated outcome. """
    results = read_control_results()
    outf = open('paper/tables/words.tex', 'w')
    outf.write('\\begin{figure*}\n\\centering\n')
    for label, noctl, ctl_only, both in sort_by_improvement(results)[:12]:
        words = read_tsv(path + 'ridge_control_and_feats/counties.norm=user.liwc.perma.pkl.out/' + label + '/words.txt', ['var', 'coef', 'f', 'alpha'])
        for w in words:
            w['r'] = math.copysign(f2r(w['f'], 100), w['coef'])
            bounds = r_ci(w['r'], 100)
            w['r_lb'] = bounds[0]
            w['r_ub'] = bounds[1]
            w['r_err'] = bounds[1] - w['r']
        top_words = sorted(words, key=lambda x: -math.fabs(x['r']))[:10]
        print '\n\n', label
        print '\n'.join('%s\t%.3f\t%.3f\t%.3f' % (w['var'], w['r_lb'], w['r'], w['r_ub']) for w in top_words)
        print 'nwords', len(words)
        plot_words(top_words, label, len(words))
        write_subfig(label, outf)
    outf.write('\\caption{For the top 12 outcomes in Table \\ref{tab.outcomes}, we plot the 10 variables with the highest correlation (error bars denote the 95\% confidence interval.) Statistical significance is indicated by $\\circ = 0.1$, $\\ast = 0.05$, $\\ast\\ast = 0.01$, $\\ast\\ast\\ast = 0.001$ (degress of freedom = 98). The thresholds have been Bonferroni-corrected using the total number of variables (160) times the number of outcomes (27). The prefix {\\sl d=} denotes lexical categories from the description field of a user\'s Twitter profile. Otherwise, the categories are derived from the tweet text. For comparison, the control variables are also included. \\label{fig.words} }\n \\end{figure*}')


def read_top_words(filename):
    """ Read text file in format: category word1 word2... (ordered by
    frequency)"""
    fp = codecs.open(filename, 'r', 'utf8')
    words = {}
    for line in fp:
        parts = line.strip().split()
        words[parts[0]] = parts[1:]
    return words


def pbox(t, w):
    """ pbox allows line breaks in table cells """
    return '\\pbox{' + w + '}{' + t + '}'


def format_outcomes(outcomes, doutcomes):
    """ Format the outcomes cell for the lexicon table """
    all_outcomes = defaultdict(lambda: [])
    for o in outcomes + doutcomes:
        all_outcomes[o['label_pp']].append(o)
    toprint = []
    for lab in sorted(all_outcomes):
        cats = all_outcomes[lab]
        s = cats[0]['label_pp'] + ':' + cats[0]['prefix'] + cats[0]['sign']
        if len(cats) > 1:
            s += '/' + cats[1]['prefix'] + cats[1]['sign']
        toprint.append(s)
    return ' '.join(sorted(toprint))


def find_similar(fips, data, fields):
    sims = [(cty, distance.cosine(np.array(data[cty])[fields], np.array(data[fips])[fields])) for cty in data if cty != fips]
    return min(sims, key=lambda x: x[1])


def lookup_f(words, word):
    for w in words:
        if w['var'] == word:
            return w['f']


def vec_diff(data, county1, county2, words):
    """ Compute the difference between the data vectors for these two
    counties, then return a list of <featurename, diff, val1, val2> tuples,
    sorted by absolute diff value """
    v1 = np.array(data.features[county1])
    v2 = np.array(data.features[county2])
    diff = np.abs(v1 - v2) / (v1 + v2)
    srt_ind = np.argsort(diff)[::-1]
    data_alpha = sorted(data.alpha.keys(), key=lambda k: data.alpha[k])
    coefs = [abs(lookup_f(words, data_alpha[i])) for i in srt_ind]
    res = zip([data_alpha[i] for i in srt_ind], coefs * diff[srt_ind],
              v1[srt_ind], v2[srt_ind])
    return sorted(res, key=lambda x: -x[1])


def error_analysis():
    """ Print information about linguistic variables that help for certain
    counties. """

    data = twdata.read(path + 'counties.norm=user.liwc.perma.pkl')
    targets, target_alpha = twexpt.read_targets('/data/twcounty/targets.tsv', data.features.keys())
    states = twexpt.read_states('/data/twcounty/states.tsv', data.features.keys())
    state_cv = twexpt.make_state_cv(data.features, states, 5)
    county_indices = [item for f in state_cv for item in f[1]]
    county_ids = np.array(sorted(data.features))[county_indices]
    control_indices = [target_alpha.index(c) for c in control_labels]
    for lab in labels:  # ['nosocialemotionalsupport', 'limitedaccesstohealthyfoods', 'ambulatorycaresensitiveconditions', 'unemployed']:
        emot_both = pickle.load(open(path + 'ridge_control_and_feats/counties.norm=user.liwc.perma.pkl.out/' + lab + '/test_folds.pkl', 'rb'))
        emot_ctl = pickle.load(open(path + 'ridge_control_only/counties.norm=none.unigrams.pkl.out/' + lab + '/test_folds.pkl', 'rb'))
        y = np.array([item for t in emot_ctl for item in t[0]])
        ctl_pred = np.array([item for t in emot_ctl for item in t[1]])
        both_pred = np.array([item for t in emot_both for item in t[1]])
        maxi = np.argmax((np.abs(ctl_pred - y) - np.abs(both_pred - y)) * (y != 0))
        #maxi = np.argmax(np.abs(ctl_pred - y) / y - np.abs(both_pred - y) / y)
        county_id = county_ids[maxi]
        print '\n', lab, '\nmost improved county is', county_id, '=', y[maxi], 'ctl=', ctl_pred[maxi], 'both=', both_pred[maxi]
        print 'this county controls=', np.array(targets[county_id])[control_indices]
        most_sim = [county_ids[np.argsort(np.abs(ctl_pred - ctl_pred[maxi]) - np.abs(y - y[maxi]) * (y != 0))[1]]]
        #most_sim = find_similar(county_id, targets, [target_alpha.index(i) for i in control_labels])
        most_sim_idx = np.where(county_ids == most_sim[0])
        print 'most sim preds=', most_sim[0], '=', y[most_sim_idx], 'ctl=', ctl_pred[most_sim_idx], 'both=', both_pred[most_sim_idx]
        print 'most sim controls=', np.array(targets[most_sim[0]])[control_indices]
        words = read_tsv(path + 'ridge_control_and_feats/counties.norm=user.liwc.perma.pkl.out/' + lab + '/words.txt', ['var', 'coef', 'f', 'alpha'])
        print 'top vector diffs=', vec_diff(data, county_id, most_sim[0], words)[:10]
        plt.plot(y, ctl_pred, 'yo')
        plt.plot(y, both_pred, 'bo')
        plt.title(lab)
        plt.xlabel('truth')
        plt.ylabel('predicted - truth')


def read_data_and_targets():
    """ Read variables and labeles from pickled data file and targets file.
    Return X, Y, data alphabet (minus control variables) and target_alphabet.
    """
    county_data = twdata.read(path + 'counties.norm=user.liwc.perma.pkl')
    counties = county_data.features
    data_alpha = sorted(county_data.alpha.keys(), key=lambda k: county_data.alpha[k])
    nonctl_alpha = copy.copy(data_alpha)
    control_names = '< 18,65 and over,Female,Afro-hispanic,med_income'
    targets, target_alpha = twexpt.read_targets('/data/twcounty/targets.tsv', counties.keys())
    twexpt.add_controls(control_names, counties, data_alpha, targets, target_alpha, False)
    control_names = [unicode(c) for c in control_names.split(',')]
    X, Y = twexpt.to_matrices(counties, targets)
    return X, Y, nonctl_alpha, target_alpha


def run_ols(vari, var, X, y, nonctl_alpha, label_pp, sig_cats):
    xx = np.c_[(X[:, vari], X[:, len(nonctl_alpha):])]
    m = sm.OLS(y, xx)
    results = m.fit()
    pval = results.pvalues[0]
    ps = pstar(pval, len(nonctl_alpha) * len(labels))  # 27=number of outcomes that are not ctl
    if ps.count('*') > 0:
        prefix = 'd' if var[:2] == 'd=' else 't'
        var = re.sub('d=', '', var)
        sig_cats[var].append({'label_pp': label_pp,
                              'prefix': prefix,
                              'sign': '+' if results.params[0] > 0 else '-'})


def run_tsls(vari, var, X, y, nonctl_alpha, label_pp, sig_cats, weights):
    """ Run two-staged least squares spatial regression and print return significant terms. """
    xx = np.c_[(X[:, vari], X[:, len(nonctl_alpha):])]
    results = twspatial.spatial_regression(xx, y, weights)
    pval = results[1][1]  # first value is intercept.
    ps = pstar(pval, len(nonctl_alpha) * len(labels))  # 3 * 160 (number of outcomes that are not ctl
    if ps.count('*') > 0:
        print label_pp, var, 'p=', pval, 'beta=', results[0][1], 'size=', len(results[1]), 'pvals=', results[1], 'betas=', results[0]
        print results[2]
        prefix = 'd' if var[:2] == 'd=' else 't'
        var = re.sub('d=', '', var)
        sig_cats[var].append({'label_pp': label_pp,
                              'prefix': prefix,
                              'sign': '+' if results[0][1] > 0 else '-'})


def make_distance_weights():
    """ Compute a distance weight matrix based on the geographical coordinates
    of each county. """
    coords = twexpt.read_coords('/data/twcounty/stats2/counties/counties.top100.bounding.txt')
    coords = [coords[cty] for cty in sorted(coords)]
    weights = pysal.Kernel(np.array(coords))
    for i in range(len(weights.weights)):
        idx = weights.neighbors[i].index(i)
        del weights.weights[i][idx]
        del weights.neighbors[i][idx]
    weights.transform = 'r'
    return weights


def find_significant_categories():
    """ Find variables that are significant predictors of outcomes using OLS
    and control variables. The idea is to only return variables that aren't
    simply reproducing control variables. """
    X, Y, nonctl_alpha, target_alpha = read_data_and_targets()
    weights = make_distance_weights()
    sig_cats = defaultdict(lambda: [])
    for labeli, label in enumerate(target_alpha):
        if twexpt.fmt_target(label) not in labels:  # Control label
            continue
        label_pp = label_map[twexpt.fmt_target(label)]
        yy = Y[:, labeli]
        # Run regression for each variable + control variables and save significant results in sig_cats.
        for vari, var in enumerate(nonctl_alpha):
            run_tsls(vari, var, X, yy, nonctl_alpha, label_pp, sig_cats, weights)
    return sig_cats


def censor(s):
    """ Censor profanity. """
    return re.sub('fucking', 'fu**ing', s)


def tab_ols():
    """ Run OLS on each variable, using control variables. The significant
    variables will then be those that aren't just surrogates for the control
    variables. """
    sig_cats = find_significant_categories()
    sig_outcomes = set()
    for v in sig_cats.itervalues():
        sig_outcomes.update([vi['label_pp'] for vi in v])
    print 'found', len(sig_cats), 'significant variables across', len(sig_outcomes)
    top_words = read_top_words('/data/twcounty/lexicons/top_words.txt')
    outf = open('paper/tables/lexicon2.tex', 'w')
    outf.write('\\begin{table*}[t]\n\\centering\n\\begin{tabular}{|r|p{5cm}|p{11cm}|}\n\hline\n')
    outf.write('%s  &  %s  &  %s\\\\\n\\hline\n' % (b('Cat'), b('Examples'), b('Outcomes')))
    printed = 0
    for var in sorted(sig_cats):
        outcomes = [s for s in sig_cats[var] if s['prefix'] == 't']
        doutcomes = [s for s in sig_cats[var] if s['prefix'] == 'd']
        if len(set([x['label_pp'] for x in outcomes + doutcomes])) > 1:
            printed += 1
            outf.write('%s  &  %s  &  %s \\\\\n\\hline\n' % (
                       var,
                       ', '.join([censor(t) for t in top_words['d=' + var][:5]]),
                       format_outcomes(outcomes, doutcomes)))

    outf.write('\\end{tabular}\n')
    outf.write('\\caption{A summary of ' + str(printed) + ' of the 80 lexical categories. These were selected by collecting all categories that are significantly correlated with at least two outcomes after controlling for demographics variables ($p < 0.05$, Bonferroni-corrected). We list the significantly correlated outcomes, the sign of correlation, and the field where the word was found: {\\sl t} for text and {\\sl d} for user description.  E.g., the second row indicates that the presence of a word from the Family category in a user description is positively correlated with teen birth rates. {\\label{tab.lexicon2}}}\n')
    outf.write('\\end{table*}\n')
    outf.close()


def fig_scatter():
    """ Print scatter plots for a few select outcomes. """
    outcomes = ['uninsured', 'teenbirthrate', 'obese']
    data = twdata.read(path + 'counties.norm=user.liwc.perma.pkl')
    targets, target_alpha = twexpt.read_targets('/data/twcounty/targets.tsv', data.features.keys())
    states = twexpt.read_states('/data/twcounty/states.tsv', data.features.keys())
    state_cv = twexpt.make_state_cv(data.features, states, 5)
    county_indices = [item for f in state_cv for item in f[1]]
    county_ids = np.array(sorted(data.features))[county_indices]

    for outcome in outcomes:
        folds = pickle.load(open(path + 'ridge/counties.norm=user.liwc.perma.pkl.out/' + outcome + '/test_folds.pkl', 'rb'))
        y = [item for t in folds for item in t[0]]
        pred = [item for t in folds for item in t[1]]
        if outcome == 'teenbirthrate':  # this is in thousands, not hundredths
            y = np.array(y) / 10.
            pred = np.array(pred) / 10.
        fit = np.polyfit(y, pred, 1)
        print 'fit=', fit
        fit_fn = np.poly1d(fit)
        # plt.scatter(y, pred)
        plt.plot(y, pred, 'bo', y, fit_fn(y), 'k')
        plt.xlabel('truth (%)', size='16')
        plt.ylabel('predicted (%)', size='16')
        plt.title(label_map[outcome], size='20')
        worst = np.argmax(abs(np.array(y) - np.array(pred)))
        county_id_worst = county_ids[worst]
        best = np.argmin(abs(np.array(y) - np.array(pred)))
        county_id_best = county_ids[best]
        max_pred = np.argmax(pred)
        county_id_max = county_ids[max_pred]
        #i = np.argmax(abs((np.array(y) * fit[0] + fit[1]) - np.array(pred)))
        print 'worst: cty=', county_id_worst, 'tr=', y[worst], 'pr=', pred[worst]
        print 'best: cty=', county_id_best, 'tr=', y[best], 'pr=', pred[best]
        print 'max: cty=', county_id_max, 'tr=', y[max_pred], 'pr=', pred[max_pred]
        words = read_tsv(path + 'ridge/counties.norm=user.liwc.perma.pkl.out/' + outcome + '/words.txt', ['var', 'coef', 'f', 'alpha'])
        print 'top vector diffs=', vec_diff(data, county_id_worst, county_id_best, words)[:10]
        #plt.show()
        plt.savefig('paper/figs/scatter_' + outcome + '.pdf', bbox_inches='tight')
        plt.close()


def main():
    tab_ols()  # Table 4
    fig_scatter()  # Figure 2
    error_analysis()  # To help make Tables 5, 6
    tab_features()  # Table 1
    tab_norm()  # Table 2
    tab_outcomes()  # Table 3
    fig_top_predictors()  # Figure 3


if __name__ == '__main__':
    main()
