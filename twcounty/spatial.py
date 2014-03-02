#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Test and correct for spatial autocorrelation when computing statistical
significance. Calls out to R to use SpatialPack
http://cran.r-project.org/web/packages/SpatialPack/index.html
"""

import argparse
import csv
import io
import json
import math
import re

import matplotlib.pyplot as plt
import numpy as np
import pysal
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import utm

import data


importr('SpatialPack')


def read_coords(filename):
    """ Return a map from county to lat/long. """
    d = {}
    for row in csv.reader(open(filename, 'rb'), delimiter='\t'):
        utms = utm.from_latlon(float(row[10]), float(row[11]))[:2]
        d[row[1]] = {'lat': float(row[10]),
                     'lon': float(row[11]),
                     'utmeast': utms[0],
                     'utmnorth': utms[1],
                     'left': float(row[12]),
                     'top': float(row[13]),
                     'right': float(row[14]),
                     'bottom': float(row[15]),
                     'state': row[0]}
    return d


def overlap(c1, c2):
    """ Return true if the bounding boxes overlap. """
    return not (c2['left'] > c1['right']
                or c2['right'] < c1['left']
                or c2['top'] > c1['bottom']
                or c2['bottom'] < c1['top'])


def intersection(c1, c2):
    return {
        'bottom': min(c1['bottom'], c2['bottom']),
        'top': max(c1['top'], c2['top']),
        'left': max(c1['left'], c2['left']),
        'right': min(c1['right'], c2['right'])}


def distance(origin, destination):
    """ Haversine. See http://www.platoscave.net/blog/2009/oct/5/calculate-distance-latitude-longitude-python/ """
    lat1, lon1 = origin
    lat2, lon2 = destination
    radius = 6371  # km

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(math.radians(lat1)) \
        * math.cos(math.radians(lat2)) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return radius * c


def get_overlaps(coords):
    """ Report how many of the given bounding boxes overlap. """
    counties = coords.keys()
    intersections = []
    for i, cty1 in enumerate(counties):
        for cty2 in counties[i + 1:]:
            if overlap(coords[cty1], coords[cty2]):
                print 'collision:', cty1, '-', coords[cty1]['state'], cty2, '-', coords[cty2]['state'], '\n', coords[cty1], '\n', coords[cty2]
                intersect = intersection(coords[cty1], coords[cty2])
                intersect['counties'] = [cty1, cty2]
                print 'Intersection=', intersect
                print 'AREA=%.3fkm\n\n' % (distance((intersect['left'], intersect['top']),
                                                    (intersect['right'], intersect['top'])) *
                                           distance((intersect['left'], intersect['top']),
                                                    (intersect['left'], intersect['bottom'])))
                intersections.append(intersect)
    return intersections


def spatial_regression(x, y, weights):
    """
    Perform Two Stage Least Square spatial regression. Weights determined using KNN. See
    http://pythonhosted.org/PySAL/library/spreg/twosls_sp.html
    >>> weights = pysal.knnW(np.array([[1, 2], [3, 1], [2, 3], [1, 2]]), 1)
    >>> r = spatial_regression([[1], [2], [3], [4]], [6, 7, 8, 9.2], weights)
    >>> r[0].T  # betas
    array([[ 4.51666667,  1.1       ,  0.04166667]])
    """
    reg = pysal.spreg.GM_Lag(np.reshape(y, (len(y), 1)), np.array(x), w=weights, w_lags=1)
    pvals = [p[1] for p in pysal.spreg.diagnostics_tsls.t_stat(reg)]
    return [b[0] for b in reg.betas], pvals, reg.summary
    # return [b[0] for b in reg.betas], [z[1] for z in reg.z_stat], reg.summary


def spatial_correlation(x, y, coords, convert_to_utm=False):
    """ Computes the correlation between x and y, adjusted for spatial
    autocorrelation. coords are an n x 2 matrix with the longitude/latitude of
    each point. Each point is first converted to UTM. Then calls out to the R library SpatialPack.
    >>> spatial_correlation([1, 2, 3, 4], [6, 7, 8, 9.2], [[1, 2], [3, 1], [2, 3], [1, 2]], convert_to_utm=False)
    (0.9989337121545471, 0.06485042171677298)
    """
    x_r = robjects.FloatVector(x)
    y_r = robjects.FloatVector(y)
    coords_utm = coords
    if convert_to_utm:
        for c in coords:
            u = utm.from_latlon(c[1], c[0])
            coords_utm.append((u[1], u[0]))
            coords_utm = [utm.from_latlon(c[0], c[1])[:2] for c in coords]
    # This, my friends, is how you convert a Python list of lists into an R matrix.
    coords_r = robjects.r.matrix(robjects.FloatVector(np.array(coords_utm).ravel()), nrow=len(coords), byrow=True)
    mttest_r = robjects.r('modified.ttest')
    results_r = mttest_r(x_r, y_r, coords_r, nclass=4)
    corr = results_r.rx('corr')[0][0]
    pval = results_r.rx('p.value')[0][0]
    return corr, pval


def read_targets(filename, counties):
    targets = dict()
    reader = csv.DictReader(open(filename), delimiter='\t')
    target_alpha = [v for v in reader.fieldnames if v != 'FIPS']
    for line in reader:
        fips = line['FIPS']
        del line['FIPS']
        if fips in counties:
            targets[fips] = line
    return targets, target_alpha


def plot_coords(coords):
    xs = []
    ys = []
    for cty in coords:
        d = coords[cty]
        plt.annotate(d['state'], xy=(d['lon'], d['lat']))
        ys.append(d['lat'])
        xs.append(d['lon'])
    plt.xlim(min(xs), max(xs))
    plt.ylim(min(ys), max(ys))
    plt.show()


def print_tweets_in_overlaps(overlaps, path, tweets):
    """ Print tweets contained by the provided list of overlap bounding boxes. """
    ids = set()
    for line in io.open(tweets, mode='rt', encoding='utf8'):
        ids.add(line.split()[1])
    counties = set()
    for o in overlaps:
        counties |= set(o['counties'])
    print 'looking up counties', counties
    matches = set()
    for cty in counties:
        path + '/' + cty
        fp = io.open(path + '/' + cty, mode='rt', encoding='utf8')
        for line in fp:
            try:
                line = re.sub('\t', '     ', line)
                js = json.loads(line)
                if valid_line(js) and str(js['id']) in ids:
                    ids.add(js['id'])
                    lon, lat = js['coordinates']['coordinates']
                    for boxes in overlaps:
                        if lon > boxes['top'] and lon < boxes['bottom'] and \
                                lat > boxes['left'] and lat < boxes['right']:
                            matches.add(js['id'])
            except:
                pass
    print 'found', len(matches), 'matches'


def valid_line(js):
    return 'id' in js and 'coordinates' in js and 'coordinates' in js['coordinates']


def main():
    """ Reports the spatial autocorrelation for each target variable. """
    ap = argparse.ArgumentParser()
    ap.add_argument('--coords', default='/data/twcounty/stats2/counties/counties.top100.bounding.txt', help='tsv file with counties, lat/long, and bounding boxes')
    ap.add_argument('--dbf', default='/data/twcounty/stats2/counties/census_polygons/gz_2010_us_050_00_500k.dbf', help='county dbf files')
    ap.add_argument('--input', default='/data/twcounty/features/counties.norm=none.liwc.pkl', help='pickled county feature Data object')
    ap.add_argument('--json', default='/data/twcounty/json', help='directory of jsons per county')
    ap.add_argument('--shapes', default='/data/twcounty/stats2/counties/census_polygons/gz_2010_us_050_00_500k.shp', help='county shape files')
    ap.add_argument('--targets', default='/data/twcounty/targets.tsv', help='targets per county')
    ap.add_argument('--tweets', default='/data/twcounty/tweets.tsv', help='tweets')
    args = ap.parse_args()

    coords = read_coords(args.coords)
    overlaps = get_overlaps(coords)
    print_tweets_in_overlaps(overlaps, args.json, args.tweets)
    shapes = pysal.open(args.shapes)
    print 'read', len(shapes), 'counties from shape file'
    dbf = pysal.open(args.dbf)
    counties = data.read(args.input).features
    targets, target_alpha = read_targets(args.targets, counties)
    for t in targets:
        didx, d = [(i, dd) for i, dd in enumerate(dbf) if dd[1] + dd[2] == t][0]
        targets[t]['dbf'] = d
        targets[t]['dbfi'] = didx
        shape = shapes[didx]
        targets[t]['shape'] = shape

    fips = sorted(targets)
    print '\t'.join(['mi.I  ', 'mi.EI  ', 'p_norm ', 'outcome'])
    weights = pysal.Kernel(np.array([targets[f]['shape'].centroid for f in fips]))
    for outcome in target_alpha:
        y = np.array([float(targets[t][outcome]) for t in fips])
        y = y / np.sum(y)
        mi = pysal.Moran(y, weights)
        print '%.5f\t%.5f\t%.5f\t%s' % (mi.I, mi.EI, mi.p_norm, outcome)


if __name__ == '__main__':
    main()
