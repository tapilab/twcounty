#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Store the word features and alphabet for a set of tweets."""

import pickle


def write(filename, data):
    pickle.dump(data, open(filename, 'wb'))


def read(filename):
    return pickle.load(open(filename, 'rb'))


class Data(object):

    def __init__(self, arguments, features, alpha):
        self.arguments = arguments
        self.features = features
        self.alpha = alpha
