# This sequence of commands should reproduce all tables and figures from the
# CHI paper, assuming everything is installed properly and the data exists in
# /data/twcounty/json/*.json

# Convert tweet jsons to one tsv.
twcounty-json2tsv /data/twcounty/json/* > /data/twcounty/tweets.tsv

# Compute stats of the data.
mkdir -p /data/twcounty/stats
twcounty-tsv2stats /data/twcounty/tweets.tsv paper/figs/

# Print top words in each lexical category.
python twcounty/top_lexicon_words.py --input /data/twcounty/tweets.tsv > /data/twcounty/stats/lexicons.txt

# Create different types of features.

# We consider two feature sets: LIWC lexicon, PERMA lexicon.
# These are extracted from both the tweet text and profile description.

# Additionally, we consider 4 feature value representations:
# - frequency of feature i (--norm none)
# - log of frequency of feature i (--norm none --log)
# - frequency of feature i divided by total tokens in a county (--norm word)
# - proportion of users in a county who used this word (--norm user)

mkdir -p /data/twcounty/features

# liwc
twcounty-tsv2feats --input /data/twcounty/tweets.tsv --min-df 20 --norm none --liwc --output /data/twcounty/features/counties.norm=none.liwc.pkl
twcounty-tsv2feats --input /data/twcounty/tweets.tsv --min-df 20 --norm user --liwc --output /data/twcounty/features/counties.norm=user.liwc.pkl
twcounty-tsv2feats --input /data/twcounty/tweets.tsv --min-df 20 --norm word --liwc --output /data/twcounty/features/counties.norm=word.liwc.pkl
twcounty-tsv2feats --input /data/twcounty/tweets.tsv --min-df 20 --norm none --liwc --log --output /data/twcounty/features/counties.norm=none.liwc.log.pkl

# perma
twcounty-tsv2feats --input /data/twcounty/tweets.tsv --min-df 20 --norm none --perma --output /data/twcounty/features/counties.norm=none.perma.pkl
twcounty-tsv2feats --input /data/twcounty/tweets.tsv --min-df 20 --norm user --perma --output /data/twcounty/features/counties.norm=user.perma.pkl
twcounty-tsv2feats --input /data/twcounty/tweets.tsv --min-df 20 --norm word --perma --output /data/twcounty/features/counties.norm=word.perma.pkl
twcounty-tsv2feats --input /data/twcounty/tweets.tsv --min-df 20 --norm none --perma --log --output /data/twcounty/features/counties.norm=none.perma.log.pkl

# liwc + perma
twcounty-tsv2feats --input /data/twcounty/tweets.tsv --min-df 20 --norm none --liwc --perma --output /data/twcounty/features/counties.norm=none.liwc.perma.pkl
twcounty-tsv2feats --input /data/twcounty/tweets.tsv --min-df 20 --norm user --liwc --perma --output /data/twcounty/features/counties.norm=user.liwc.perma.pkl
twcounty-tsv2feats --input /data/twcounty/tweets.tsv --min-df 20 --norm word --liwc --perma --output /data/twcounty/features/counties.norm=word.liwc.perma.pkl
twcounty-tsv2feats --input /data/twcounty/tweets.tsv --min-df 20 --norm none --liwc --perma --log --output /data/twcounty/features/counties.norm=none.liwc.perma.log.pkl

# Run prediction experiments.

# Ridge regression on all feature sets
for pkl in /data/twcounty/features/*.pkl; do \
    echo $pkl; \
    twcounty-expt --model ridge --states /data/twcounty/states.tsv --targets /data/twcounty/targets.tsv --input $pkl
done;
mkdir -p /data/twcounty/features/ridge
mv /data/twcounty/features/*out /data/twcounty/features/ridge

# Ridge regression using control variables + features
for pkl in /data/twcounty/features/*.pkl; do \
    echo $pkl; \
    twcounty-expt --model ridge --states /data/twcounty/states.tsv --targets /data/twcounty/targets.tsv --controls '< 18,65 and over,Female,Afro-hispanic,med_income' --input $pkl
done;
mkdir -p /data/twcounty/features/ridge_control_and_feats
mv /data/twcounty/features/*out /data/twcounty/features/ridge_control_and_feats

# Ridge regression using control variables alone
twcounty-expt --model ridge --states /data/twcounty/states.tsv --targets /data/twcounty/targets.tsv --controls '< 18,65 and over,Female,Afro-hispanic,med_income' --input /data/twcounty/features/counties.norm=none.unigrams.pkl --controls-only
mkdir -p /data/twcounty/features/ridge_control_only
mv /data/twcounty/features/*out /data/twcounty/features/ridge_control_only

# Print top words per lexicon item
python twcounty/top_lexicon_words.py --input /data/twcounty/tweets.tsv > /data/twcounty/lexicons/top_words.txt

# Create tables and figures
python -m twcounty.texify_results
