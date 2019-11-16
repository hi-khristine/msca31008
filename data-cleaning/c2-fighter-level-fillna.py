# author: Bryce C working off code from Khristine.
# purpose: handle NAs for fighter-level data.
# note: 
#   this code should be run from the data-cleaning directory.
#   when it runs, you'll have a corrected CSV file in the out/ folder
#   called c2-figher-level-transform.csv

import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# read in functions.
import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

fighter_dataset = pd.read_csv( '../out/c1-fighter-level-transform.csv' )

num_dat = fighter_dataset.select_dtypes(include=np.number) #extract numeric columns


# Fill in missing values using kmeans

from sklearn.cluster import KMeans

def kmeans_missing(X, n_clusters, max_iter=10):
    """Perform K-Means clustering on data with missing values.

    Args:
      X: An [n_samples, n_features] array of data to cluster.
      n_clusters: Number of clusters to form.
      max_iter: Maximum number of EM iterations to perform.

    Returns:
      labels: An [n_samples] vector of integer labels.
      centroids: An [n_clusters, n_features] array of cluster centroids.
      X_hat: Copy of X with the missing values filled in.
    """

    # Initialize missing values to their column means
    missing = ~np.isfinite(X)
    mu = np.nanmean(X, 0, keepdims=1)
    X_hat = np.where(missing, mu, X)

    for i in range(max_iter):
        if i > 0:
            # initialize KMeans with the previous set of centroids. this is much
            # faster and makes it easier to check convergence (since labels
            # won't be permuted on every iteration), but might be more prone to
            # getting stuck in local minima.
            cls = KMeans(n_clusters, init=prev_centroids)
        else:
            # do multiple random initializations in parallel
            cls = KMeans(n_clusters, n_jobs=-1)

        # perform clustering on the filled-in data
        labels = cls.fit_predict(X_hat)
        centroids = cls.cluster_centers_

        # fill in the missing values based on their cluster centroids
        X_hat[missing] = centroids[labels][missing]

        # when the labels have stopped changing then we have converged
        if i > 0 and np.all(labels == prev_labels):
            break

        prev_labels = labels
        prev_centroids = cls.cluster_centers_

    return labels, centroids, X_hat


#num_dat.columns[num_dat.isna().any()].tolist()
na_num_cols= ['age',
 'Height_cms',
 'Reach_cms',
 'Weight_lbs',
 'avg_BODY_att',
 'avg_BODY_landed',
 'avg_CLINCH_att',
 'avg_CLINCH_landed',
 'avg_DISTANCE_att',
 'avg_DISTANCE_landed',
 'avg_GROUND_att',
 'avg_GROUND_landed',
 'avg_HEAD_att',
 'avg_HEAD_landed',
 'avg_KD',
 'avg_LEG_att',
 'avg_LEG_landed',
 'avg_PASS',
 'avg_REV',
 'avg_SIG_STR_att',
 'avg_SIG_STR_landed',
 'avg_SIG_STR_pct',
 'avg_SUB_ATT',
 'avg_TD_att',
 'avg_TD_landed',
 'avg_TD_pct',
 'avg_TOTAL_STR_att',
 'avg_TOTAL_STR_landed',
 'total_time_fought(seconds)',
 'avg_opp_BODY_att',
 'avg_opp_BODY_landed',
 'avg_opp_CLINCH_att',
 'avg_opp_CLINCH_landed',
 'avg_opp_DISTANCE_att',
 'avg_opp_DISTANCE_landed',
 'avg_opp_GROUND_att',
 'avg_opp_GROUND_landed',
 'avg_opp_HEAD_att',
 'avg_opp_HEAD_landed',
 'avg_opp_KD',
 'avg_opp_LEG_att',
 'avg_opp_LEG_landed',
 'avg_opp_PASS',
 'avg_opp_REV',
 'avg_opp_SIG_STR_att',
 'avg_opp_SIG_STR_landed',
 'avg_opp_SIG_STR_pct',
 'avg_opp_SUB_ATT',
 'avg_opp_TD_att',
 'avg_opp_TD_landed',
 'avg_opp_TD_pct',
 'avg_opp_TOTAL_STR_att',
 'avg_opp_TOTAL_STR_landed']

na_num_dat = num_dat[na_num_cols]


# commenting this out until we can get it running:
#labels, centroids, X_hat = kmeans_missing(fill, n_clusters=4) #I ran this for a bunch of columns individually and saw 4 was the number of clusters in all instances
#fighter_dataset[na_num_cols]= X_hat

save( '../out/c2-fighter-level-fillna.pkl', na_num_cols, fighter_dataset )
