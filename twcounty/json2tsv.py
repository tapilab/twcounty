#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Extract the tweet id, user name, text, and user description from json files.
Expects one file per county, where name of file is county id.
"""

import codecs
import io
import json
import os
import re
import sys

import tok.unicode_props

sys.stdout = codecs.getwriter('utf8')(sys.stdout)


def tokenize(s, tokenizer):
    """
    >>> tokenize("hello, mr. jone-s don't http://www.foo.com @foo yeah", tok.unicode_props.UP_Tiny(1))
    'hello mr jone s don t http thisisamention yeah'
    """
    s = re.sub('@\S+', 'thisisamention', s)  # map all mentions to thisisamention
    s = re.sub('http\S+', 'http', s)  # keep only http from urls
    return ' '.join(tokenizer.tokenize(s))


def valid_line(js):
    return 'id' in js and 'user' in js and 'text' in js and 'screen_name' in js['user'] and 'description' in js['user']


def parse(f):
    """ Parse a json tweet file and print the tokenized result to stdout. """
    tokenizer = tok.unicode_props.UP_Tiny(1)
    fp = io.open(f, mode='rt', encoding='utf8')
    county = os.path.basename(f)
    for line in fp:
        try:
            line = re.sub('\t', '     ', line)
            js = json.loads(line)
            if valid_line(js):
                print '\t'.join([county,
                                 str(js['id']),
                                 js['user']['screen_name'],
                                 tokenize(js['text'], tokenizer),
                                 tokenize(js['user']['description'], tokenizer)])
        except:
            pass


def main():
    for f in sys.argv[1:]:
        parse(f)


if __name__ == '__main__':
    main()
