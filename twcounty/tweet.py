#!/usr/bin/env python
# -*- coding: utf-8 -*-

import codecs


def from_tsv(filename):
    fp = codecs.open(filename, 'r', 'utf8')
    for line in fp:
        parts = line.strip().split('\t')
        if len(parts) > 2:
            parts.extend([None] * (5 - len(parts)))  # add blanks for missing fields
            yield Tweet(*parts)
        else:
            print 'bad tweet:', line


class Tweet(object):

    def __init__(self, county, id_, screen_name, text, description):
        self.county = county
        self.id_ = id_
        self.screen_name = screen_name
        self.text = text
        self.description = description
        self.features = []
