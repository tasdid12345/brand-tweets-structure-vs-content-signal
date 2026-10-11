# Does tweet content beat structure alone for predicting negative sentiment?

## Motivation
The paper "Study of competition in the textile sector by twitter social network analysis" (2018) studies competing brands through the structure of their Twitter networks. This experiment asks whether adding the text of tweets to such structural and engagement information gives a better view of how customers see competing brands. Public airline tweets are used as a stand-in for textile-brand data.

## Question
On public tweets about competing airline brands, does adding tweet content to structural and engagement features predict negative customer sentiment better than a structure-only model?

## Dataset
Twitter US Airline Sentiment (`Tweets.csv`, about 14,600 tweets on 6 airlines), downloaded from the Hugging Face dataset repo `osanseviero/twitter-airline-sentiment`. This is not the paper's own dataset. The target is whether a tweet is negative; one stratified 80/20 train/test split with seed 42.

## Models
- **Baseline:** LightGBM on structural/engagement features (brand, number of @mentions, hashtags, URLs, retweet count, tweet length).
- **Improved:** logistic regression on word and character TF-IDF of the tweet text stacked with the same structural features. Regularisation (C) is tuned by cross-validation on the training set only.

Both models share the same data, preprocessing, CV folds and test-set evaluation. The main metric is test ROC-AUC; accuracy and macro-F1 are secondary. Randomized models are run with seeds 42, 43 and 44.

## How to run
```
pip install -r requirements.txt
python analysis.py
```

## Results
Results are written to `results.json` (and a chart to `figures/`) when the script runs.

## Reference

This project is motivated by the following paper:

> AB Gil-González, AL Blanco-Mateos, F De la Prieta (2018). *Study of competition in the textile sector by twitter social network analysis*. https://www.researchgate.net/profile/Ana-Gil-22/publication/325466325_Study_of_Competition_in_the_Textile_Sector_by_Twitter_Social_Network_Analysis/links/5b0fbef8a6fdcc80995c16e1/Study-of-Competition-in-the-Textile-Sector-by-Twitter-Social-Network-Analysis.pdf

Paper author: Prof. Ana Belén Gil González, University of Salamanca (abg@usal.es)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
