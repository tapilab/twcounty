#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import logging.config


__author__ = 'Aron Culotta'
__email__ = 'aronwc@gmail.com'
__version__ = '0.1.0'

# create logger
logging.config.fileConfig('logging.conf')
logger = logging.getLogger(__name__)
