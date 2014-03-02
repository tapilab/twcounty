# twcounty

Code to recreate the results for the paper: "Estimating County Health
Statistics with Twitter," Aron Culotta, CHI 2014.

Note that this is not intended as a library for Twitter analysis; instead,
it's main goal is to allow other researchers to reproduce results. This is
unfortunately not that easy to do, since it relies on Twitter data (which is
hard to share) and LIWC and PERMA lexicons (which are not freely available).

## Installation

- Dependencies are listed in [`requirements.txt`](requirements.txt).
- Additionally, [json2tsv.py](twcounty/json2tsv.py) uses the tokenizer from
[QUAC](https://github.com/reidpr/quac/blob/master/lib/tok/unicode_props.py),
which as of this writing does not allow easy_install. So, you'll have to
install QUAC manually following
[these instructions](http://reidpr.github.io/quac/install.html), then add
`quac` to your `PYTHONPATH`.
- Once all dependencies are installed, run `python setup.py develop`, which
  will install some command-line aliases used by [`run.sh`](run.sh)
  (`twcounty-expt`, `twcounty-json2tsv`, `twcounty-tsv2feats`,
  `twcounty-tsv2stats`).


## Data

1. Paths to [LIWC](http://www.liwc.net/) and [PERMA](http://www.cis.upenn.edu/~ungar/CVs/WWBP.html) lexicons should be in environmental variables `$LIWC` and `$PERMA`. (Unfortunately, these are not free for download.) The expected format is:

```
%
1     Pronoun
2     Adjective
...
%
he    1
pretty     2
...
2. The code expects tweets in `/data/twcounty/json/`. Each file should be
named after the
[FIPS](http://en.wikipedia.org/wiki/Federal_Information_Processing_Standards)
of the county from which the tweets originated. The contents should be the raw
json returned by the Twitter API. As sharing raw tweets violates Twitter's
TOS, please send email to Aron Culotta (aronwc at gmail.com), and I'll send
you the list of tweet IDs. You will then have to submit queries to the Twitter
API to retrieve the raw content.

## Reproducing results

Once everything is installed and the data is in place, you should be able to
run the script `run.sh`. Results will be written to `paper/tables` and
`paper/figs`. Pickled data files and raw results will be in
`/data/twcounty/features`.
