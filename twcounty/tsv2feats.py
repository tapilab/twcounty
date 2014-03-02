#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Given a tsv of tweets produced by json2tsv, output one pickled feature vector
for each county.
"""
import argparse
from collections import defaultdict
import logging
import os
import math
import psutil
import sys

import data
import lexicon
import tweet


logger = logging.getLogger(__name__)


def add_prefix(words, prefix):
    for i in range(words):
        words[i] = prefix + words[i]


def lexicons(words, liwc, perma):
    """ Look up each word in liwc and/or perma lexicons. Lexicon location
    determined by $LIWC and $PERMA environment variables.

    >>> lexicons(['i'], True, False)
    [u'Pronoun', u'I', u'Self']
    >>> lexicons(['abject'], False, True)
    [u'P-']
    >>> lexicons(['i', 'abject'], True, True)
    [u'Pronoun', u'I', u'Self', u'P-']
    """
    result = []
    if liwc:
        for w in words:
            result.extend(lexicon.liwc.categories_for_token(w))
    if perma:
        for w in words:
            result.extend(lexicon.perma.categories_for_token(w))
    return result


def do_featurize(feats, words, prefix, alpha, unigrams=True, liwc=False, perma=False):
    lexi = lexicons(words, liwc, perma)
    if unigrams:
        lexi.extend(words)
    for t in lexi:
        feats[alpha[prefix + t]] += 1
    return feats


def featurize(tw, alpha, unigrams, liwc, perma):
    """
    >>> alpha = defaultdict(lambda: len(alpha))
    >>> feats = featurize(tweet.Tweet('cty1', '123', 'joe', 'i', 'abject'), alpha, True, True, True)
    >>> all(feats[alpha[word]] == 1 for word in [u'd=P-', u'Pronoun', u'I', u'Self', 'i', 'd=abject'])
    True
    """
    feats = defaultdict(lambda: 0)
    if tw.text:
        do_featurize(feats, tw.text.split(' '), '', alpha, unigrams, liwc, perma)
    if tw.description:
        do_featurize(feats, tw.description.split(' '), 'd=', alpha, unigrams, liwc, perma)
    return feats


def read_tweets(tweets, user_norm=False, unigrams=True, liwc=False, perma=False):
    """
    >>> tbc, ubc, alpha = read_tweets([tweet.Tweet('c1', '', 'joe', 'i', ''), tweet.Tweet('c1', '', 'joe', 'i', ''), tweet.Tweet('c1', '', 'bob', 'i', '')], True, True, False, False)
    >>> tbc['c1'][0]
    2
    """
    alpha = defaultdict(lambda: len(alpha))
    # county -> tweets
    tweets_by_county = defaultdict(lambda: defaultdict(lambda: 0))
    users_by_county = defaultdict(lambda: set())
    tweets_by_user = defaultdict(lambda: defaultdict(lambda: set()))
    i = 0
    for tw in tweets:
        users_by_county[tw.county].add(tw.screen_name)
        feats = featurize(tw, alpha, unigrams, liwc, perma)
        for k, v in feats.iteritems():
            if user_norm:
                if k not in tweets_by_user[tw.county][tw.screen_name]:
                    tweets_by_county[tw.county][k] += 1
                    tweets_by_user[tw.county][tw.screen_name].add(k)
            else:
                tweets_by_county[tw.county][k] += v
        i += 1
        if i % 100000 == 0:
            logger.info('#tweets=%d #words=%d mem=%s' % (i, len(alpha), mem_usage()))
    return default_dict_to_dict(tweets_by_county), users_by_county, alpha


def mem_usage():
    return fmt_real(psutil.Process(os.getpid()).get_memory_info().rss, 1024, ["B", "KiB", "MiB", "GiB", "TiB", "PiB"])


def fmt_real(num, factor, units):
    assert num >= 0, "negative numbers unimplemented"
    factor = float(factor)
    for unit in units:
        if (num < factor):
            return ("%.1f%s" % (num, unit))
        num /= factor
    assert False, "number too large"


def default_dict_to_dict(d):
    return dict((k, v) for k, v in d.iteritems())


def norm_by_user(tweets_by_county, users_by_county, alpha):
    """ Each county is represented by the fraction of users who use a particular word
    >>> alpha = {'dog': 0, 'cat': 1}
    >>> counties = {'cty1': {0: 2, 1: 1}}
    >>> norm_by_user(counties, {'cty1': set(['u1', 'u2'])}, alpha)
    >>> counties['cty1']
    {0: 1.0, 1: 0.5}
    """
    for cty_id in tweets_by_county:
        nusers = len(users_by_county[cty_id])
        features = tweets_by_county[cty_id]
        for feature, value in features.iteritems():
            features[feature] = 1. * value / nusers


def norm_by_word(tweets_by_county, alpha):
    """ Each county is represented by the frequency of a word divided by the
    total number of tokens in a county.
    >>> alpha = {'dog': 0, 'cat': 1}
    >>> counties = {'cty1': {0: 30, 1: 20}}
    >>> norm_by_word(counties, alpha)
    >>> counties['cty1']
    {0: 0.6, 1: 0.4}
    """
    for county in tweets_by_county.itervalues():
        total = sum(county.itervalues())
        for feature, value in county.iteritems():
            county[feature] = 1. * value / total


def normalize(tweets_by_county, users_by_county, alpha, method='user'):
    if method == 'user':
        norm_by_user(tweets_by_county, users_by_county, alpha)
    elif method == 'word':
        norm_by_word(tweets_by_county, alpha)
    elif method == 'none':
        pass
    else:
        raise ValueError('unknown norm method: ' + method)


def logify(features):
    for cty in features:
        features[cty] = [math.log(v + 1) for v in features[cty]]


def count_words_by_county(tweets_by_county):
    """ Count the number of different counties each word appears in.
    >>> counts = count_words_by_county({'cty1': {0: 1, 1: 2}, 'cty2': {0: 100}})
    >>> counts[0]
    2
    >>> counts[1]
    1
    """
    counter = defaultdict(lambda: 0)
    for county in tweets_by_county.itervalues():
        for feature in county:
            counter[feature] += 1
    return counter


def reverse_dict(alpha):
    """
    >>> reverse_dict({0: 'a', 1: 'b'})
    {'a': 0, 'b': 1}
    """
    return dict([(v, k) for k, v in alpha.iteritems()])


def filter_by_count(tweets_by_county, alpha, min_df):
    """ Remove features that dont occur in at least min_df different
    counties. Return new alphabet as well as new feature vectors.
    >>> tweets, alpha = filter_by_count({'cty1': {0: 1, 1: 2}, 'cty2': {1: 100}}, {'a': 0, 'b': 1}, 2)
    >>> 'a' in alpha
    False
    >>> 'b' in alpha
    True
    >>> tweets['cty1'][0]
    2
    >>> tweets['cty2'][0]
    100
    """
    word_counts = count_words_by_county(tweets_by_county)
    new_alpha = defaultdict()
    rev_alpha = reverse_dict(alpha)
    map_alpha = defaultdict()
    # make new alphabet
    for feat_idx, count in word_counts.iteritems():
        if count >= min_df:
            new_alpha[rev_alpha[feat_idx]] = len(new_alpha)
            map_alpha[feat_idx] = new_alpha[rev_alpha[feat_idx]]
    # update all feature vectors
    for county in tweets_by_county:
        new_fv = defaultdict(lambda: 0, [(map_alpha[fi], val) for fi, val in tweets_by_county[county].iteritems() if fi in map_alpha])
        tweets_by_county[county] = new_fv
    logger.info('pruned alpha from %d to %d' % (len(alpha), len(new_alpha)))
    return tweets_by_county, new_alpha


def densify(features, alpha):
    """ convert feature dicts to lists """
    for cty, feats in features.iteritems():
        features[cty] = [feats[f] for f in range(len(alpha))]
    return features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='tweets.tsv', help='tweets file')
    ap.add_argument('--liwc', action='store_true', default=False, help='use LIWC features [unimpl]')
    ap.add_argument('--log', action='store_true', default=False, help='compute log of feature values')
    ap.add_argument('--min-df', metavar='MIN_DF', type=int, default=20, help='min number of different county a word must appear in')
    ap.add_argument('--norm', default='none', help='how to normalize counts [user, word, none]')
    ap.add_argument('--output', default='counties.pkl', help='output file')
    ap.add_argument('--perma', action='store_true', default=False, help='use PERMA features [unimpl]')
    ap.add_argument('--unigrams', action='store_true', default=False, help='retain unigrams')
    args = ap.parse_args()
    logger.info(args)

    if os.path.exists(args.output):
        logger.info('already done!')
        sys.exit()
    features, users_by_county, alpha = read_tweets(tweet.from_tsv(args.input), args.norm == 'user', args.unigrams, args.liwc, args.perma)
    features, alpha = filter_by_count(features, alpha, args.min_df)
    normalize(features, users_by_county, alpha, args.norm)
    features = densify(features, alpha)  # default_dict_to_dict(features)
    if args.log:
        logify(features)
    logger.info('10 words from alpha: %s' % alpha.keys()[:10])
    logger.info('first 10 features from first county: %s' % features.values()[0][:10])
    data.write(args.output, data.Data(args, features, alpha))


if __name__ == '__main__':
    main()
