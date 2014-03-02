#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Print the top words for each lexical category from a tsv file.
"""
import argparse
import codecs
from collections import Counter, defaultdict
import logging
import sys

import lexicon
import tweet


logger = logging.getLogger(__name__)
sys.stdout = codecs.getwriter('utf8')(sys.stdout)


def update_counts(text, counts, lexi):
    if lexi and text:
        for word in text.split(' '):
            cats = lexi.categories_for_token(word)
            for c in cats:
                counts[c].update([word])


def count_words(filename):
    liwc = lexicon.liwc
    perma = lexicon.perma
    counts_text = defaultdict(lambda: Counter())
    counts_desc = defaultdict(lambda: Counter())
    for tw in tweet.from_tsv(filename):
        update_counts(tw.text, counts_text, liwc)
        update_counts(tw.text, counts_text, perma)
        update_counts(tw.description, counts_desc, liwc)
        update_counts(tw.description, counts_desc, perma)
    return counts_text, counts_desc


def print_top_words(counts, n=20, prefix=''):
    for cat in sorted(counts):
        print '%s%s %s' % (prefix, cat, ' '.join(('%s' % k) for k, v in counts[cat].most_common(n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='tweets.tsv', help='tweets file')
    args = ap.parse_args()
    logger.info(args)
    text, desc = count_words(args.input)
    print_top_words(text)
    print_top_words(desc, n=20, prefix='d=')

if __name__ == '__main__':
    main()
