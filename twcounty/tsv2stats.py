#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Given a tsv of tweets produced by json2tsv, output some statistics. Generates
figure 1 from the CHI paper.
"""
import argparse
import codecs
from collections import defaultdict
from operator import itemgetter
import logging

import matplotlib.pyplot as plt
import scipy.stats as scistat

import tweet


logger = logging.getLogger(__name__)


def tweets_per_county(stats):
    """
    >>> tweets_per_county({'cty1': {'u1': 10, 'u2': 90}, 'cty2': {'u3': 10000}})
    [4.0, 2.0]
    """
    return sorted((sum(stats[cty].itervalues()) for cty in stats), reverse=True)


def users_per_county(stats):
    """
    >>> users_per_county({'cty1': {'u1': 10, 'u2': 90}, 'cty2': {'u3': 10000}}) # doctest: +ELLIPSIS
    [0.3..., 0.0]
    """
    return sorted((len(stats[cty].values()) for cty in stats), reverse=True)


def tweets_per_user(stats):
    """
    >>> tweets_per_user({'cty1': {'u1': 10, 'u2': 100}, 'cty2': {'u3': 10000}})
    [4.0, 2.0, 1.0]
    """
    return sorted((stats[cty][usr] for cty in stats.iterkeys()
                  for usr in stats[cty].iterkeys()), reverse=True)


def make_plot(filename, values, xlabel, ylabel, title):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', labelsize='24')
    ax.tick_params(axis='both', pad=10)
    plt.plot(values)
    ax.set_xscale('log')
    ax.set_yscale('log')
    plt.xlabel(xlabel, size='24')
    plt.ylabel(ylabel, size='24')
    #ax.spines['left'].set_visible(False)
    plt.title(title, size='24')
    plt.savefig(filename, bbox_inches='tight')
    plt.close()


def read_county_populations(filename):
    """ Expects TSV file where the second column is FIPS (county code) and the
    fifth column is the population """
    pops = {}
    fp = codecs.open(filename, 'r', 'utf8')
    for line in fp:
        parts = line.strip().split('\t')
        #pops[parts[1]] = math.log(int(parts[4]), 10.)
        pops[parts[1]] = int(parts[4])
    return pops


def plot_population_by_users(pops, stats, filename):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.tick_params(axis='both', labelsize='24')
    ax.tick_params(axis='y', pad=10)
    userspc = dict([(cty, len(stats[cty])) for cty in stats])
    print 'correl with users:', scistat.pearsonr([pops[cty] for cty in sorted(stats)],
                                                 [userspc[cty] for cty in sorted(stats)])
    print 'correl with tweets:', scistat.pearsonr([pops[cty] for cty in sorted(stats)],
                                                  [userspc[cty] for cty in sorted(stats)])
    userspc = dict([(cty, sum(stats[cty].itervalues())) for cty in stats])
    plt.plot([pops[cty] for cty in sorted(stats)], [userspc[cty] for cty in sorted(stats)], 'bo')
    plt.xlabel('Population', size='24')
    plt.ylabel('Twitter Users', size='24')
    #ax.spines['left'].set_visible(False)
    plt.title('Twitter Users by County Population', size='24')
    plt.savefig(filename, bbox_inches='tight')


def print_top_bot_counties(stats):
    counts = [(cty, len(stats[cty].values())) for cty in stats]
    maxx = max(counts, key=itemgetter(1))
    minn = min(counts, key=itemgetter(1))
    print 'biggest county is', maxx[0], 'with', maxx[1], 'users'
    print 'smallest county is', minn[0], 'with', minn[1], 'users'
    print 'fraction above 10k is', 1. * len([c for c in counts if c[1] >= 10000]) / len(counts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='/data/twcounty/tweets.tsv', help='tweets file')
    ap.add_argument('--output', default='paper/figs', help='output dir')
    ap.add_argument('--pop', default='/data/twcounty/counties.tsv', help='county population data: <state> <FIPS> <otherid> <name> <pop>')
    args = ap.parse_args()
    logger.info(args)

    stats = defaultdict(lambda: defaultdict(lambda: 0))
    outd = args.output
    for tw in tweet.from_tsv(args.input):
        stats[tw.county][tw.screen_name] += 1
    pops = read_county_populations(args.pop)
    plot_population_by_users(pops, stats, outd + '/users_by_pop.pdf')
    print_top_bot_counties(stats)
    tpc = tweets_per_county(stats)
    upc = users_per_county(stats)
    tpu = tweets_per_user(stats)
    make_plot(outd + '/tweets_per_county.pdf', tpc, 'Rank', 'Count', 'Tweets Per County')
    make_plot(outd + '/users_per_county.pdf', upc, 'Rank', 'Count', 'Users Per County')
    make_plot(outd + '/tweets_per_user.pdf', tpu, 'Rank', 'Count', 'Tweets Per User')
    print 'ntweets=', sum(v for v in tpc)
    print 'nusers=', sum(v for v in upc)
    print 'ncounties=', len(tpc)

if __name__ == '__main__':
    main()
