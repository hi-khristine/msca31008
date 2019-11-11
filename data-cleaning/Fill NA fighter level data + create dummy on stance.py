#!/usr/bin/env python
# coding: utf-8

# In[32]:


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# In[33]:


data=pd.read_csv("data.csv")
data.head()


# ADAM CODE START HERE
# -----

# In[34]:


for i in range(len(data.columns)):
    print(data.columns[i])

all_fighters = list(data['R_fighter'].append(data['B_fighter']).unique())

## Categorize Columns
fight_columns = ['Referee', 'date', 'location', 'Winner', 'title_bout', 'weight_class', 'no_of_rounds']

red_fighter_columns = ['R_fighter', 'R_age', 'R_Height_cms', 'R_Reach_cms', 'R_Weight_lbs', 'R_Stance']
red_stats_columns = ['R_avg_BODY_att', 'R_avg_BODY_landed', 'R_avg_CLINCH_att', 'R_avg_CLINCH_landed', 'R_avg_DISTANCE_att', 'R_avg_DISTANCE_landed', 'R_avg_GROUND_att', 'R_avg_GROUND_landed', 'R_avg_HEAD_att', 'R_avg_HEAD_landed', 'R_avg_KD', 'R_avg_LEG_att', 'R_avg_LEG_landed', 'R_avg_PASS', 'R_avg_REV', 'R_avg_SIG_STR_att', 'R_avg_SIG_STR_landed', 'R_avg_SIG_STR_pct', 'R_avg_SUB_ATT', 'R_avg_TD_att', 'R_avg_TD_landed', 'R_avg_TD_pct', 'R_avg_TOTAL_STR_att', 'R_avg_TOTAL_STR_landed']
red_history_columns = ['R_current_lose_streak', 'R_current_win_streak', 'R_draw', 'R_longest_win_streak', 'R_losses', 'R_total_rounds_fought', 'R_total_time_fought(seconds)', 'R_total_title_bouts', 'R_win_by_Decision_Majority', 'R_win_by_Decision_Split', 'R_win_by_Decision_Unanimous', 'R_win_by_KO/TKO', 'R_win_by_Submission', 'R_win_by_TKO_Doctor_Stoppage', 'R_wins']
red_opp_stats_columns = ['R_avg_opp_BODY_att', 'R_avg_opp_BODY_landed', 'R_avg_opp_CLINCH_att', 'R_avg_opp_CLINCH_landed', 'R_avg_opp_DISTANCE_att', 'R_avg_opp_DISTANCE_landed', 'R_avg_opp_GROUND_att', 'R_avg_opp_GROUND_landed', 'R_avg_opp_HEAD_att', 'R_avg_opp_HEAD_landed', 'R_avg_opp_KD', 'R_avg_opp_LEG_att', 'R_avg_opp_LEG_landed', 'R_avg_opp_PASS', 'R_avg_opp_REV', 'R_avg_opp_SIG_STR_att', 'R_avg_opp_SIG_STR_landed', 'R_avg_opp_SIG_STR_pct', 'R_avg_opp_SUB_ATT', 'R_avg_opp_TD_att', 'R_avg_opp_TD_landed', 'R_avg_opp_TD_pct', 'R_avg_opp_TOTAL_STR_att', 'R_avg_opp_TOTAL_STR_landed']

blue_fighter_columns = ['B_fighter', 'B_age', 'B_Height_cms', 'B_Reach_cms', 'B_Weight_lbs', 'B_Stance']
blue_stats_columns = ['B_avg_BODY_att', 'B_avg_BODY_landed', 'B_avg_CLINCH_att', 'B_avg_CLINCH_landed', 'B_avg_DISTANCE_att', 'B_avg_DISTANCE_landed', 'B_avg_GROUND_att', 'B_avg_GROUND_landed', 'B_avg_HEAD_att', 'B_avg_HEAD_landed', 'B_avg_KD', 'B_avg_LEG_att', 'B_avg_LEG_landed', 'B_avg_PASS', 'B_avg_REV', 'B_avg_SIG_STR_att', 'B_avg_SIG_STR_landed', 'B_avg_SIG_STR_pct', 'B_avg_SUB_ATT', 'B_avg_TD_att', 'B_avg_TD_landed', 'B_avg_TD_pct', 'B_avg_TOTAL_STR_att', 'B_avg_TOTAL_STR_landed']
blue_history_columns= ['B_current_lose_streak', 'B_current_win_streak', 'B_draw', 'B_longest_win_streak', 'B_losses', 'B_total_rounds_fought', 'B_total_time_fought(seconds)', 'B_total_title_bouts', 'B_win_by_Decision_Majority', 'B_win_by_Decision_Split', 'B_win_by_Decision_Unanimous', 'B_win_by_KO/TKO', 'B_win_by_Submission', 'B_win_by_TKO_Doctor_Stoppage', 'B_wins']
blue_opp_stats_columns = ['B_avg_opp_BODY_att', 'B_avg_opp_BODY_landed', 'B_avg_opp_CLINCH_att', 'B_avg_opp_CLINCH_landed', 'B_avg_opp_DISTANCE_att', 'B_avg_opp_DISTANCE_landed', 'B_avg_opp_GROUND_att', 'B_avg_opp_GROUND_landed', 'B_avg_opp_HEAD_att', 'B_avg_opp_HEAD_landed', 'B_avg_opp_KD', 'B_avg_opp_LEG_att', 'B_avg_opp_LEG_landed', 'B_avg_opp_PASS', 'B_avg_opp_REV', 'B_avg_opp_SIG_STR_att', 'B_avg_opp_SIG_STR_landed', 'B_avg_opp_SIG_STR_pct', 'B_avg_opp_SUB_ATT', 'B_avg_opp_TD_att', 'B_avg_opp_TD_landed', 'B_avg_opp_TD_pct', 'B_avg_opp_TOTAL_STR_att', 'B_avg_opp_TOTAL_STR_landed']

## Pull data for red fighters. 0 indicates a draw, which is a value in the Winner column.
red_fighters = data[fight_columns + red_fighter_columns + red_stats_columns + red_history_columns + red_opp_stats_columns]
red_fighters['Win'] = red_fighters['Winner'].apply(lambda x: 1 if x == 'Red' else (-1 if x == 'Blue' else 0))

## Pull corresponding data for blue fighters. 0 indicates a draw.
blue_fighters = data[fight_columns + blue_fighter_columns + blue_stats_columns + blue_history_columns + blue_opp_stats_columns]
blue_fighters['Win'] = blue_fighters['Winner'].apply(lambda x: 1 if x == 'Blue' else (-1 if x == 'Red' else 0))


## Match the column names
red_fighters.rename(columns = lambda x: x.strip('R_'), inplace = True)
red_fighters.rename(columns = {'eferee': 'Referee', 'each_cms': 'Reach_cms'}, inplace = True)
blue_fighters.rename(columns = lambda x: x.strip('B_'), inplace = True)


# In[35]:


fighter_dataset = red_fighters.append(blue_fighters)

fighter_dataset.sort_values(by = ['fighter', 'date'], inplace = True)


# KHRISTINE CODE START HERE
# ------

# In[36]:


num_dat=fighter_dataset.select_dtypes(include=np.number) #extract numeric columns


# Fill in missing values using kmeans

# In[37]:


import numpy as np
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


# In[38]:


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


# In[39]:


labels, centroids, X_hat = kmeans_missing(fill, n_clusters=4) #I ran this for a bunch of columns individually and saw 4 was the number of clusters in all instances


# In[40]:


fighter_dataset[na_num_cols]= X_hat


# Create dummy columns for Stance

# In[41]:


fighter_dataset=pd.get_dummies(fighter_dataset, columns=['Stance'])


# In[43]:


fighter_dataset.to_csv('fighter_level_dataset_NAFILL_DUMMY.csv')


# In[ ]:


##Standardize data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


stscaler=scaler.fit(fighter_dataset[na_num_cols])

fighter_dataset[na_num_cols] = stscaler.transform(fighter_dataset[na_num_cols])


##Train Test Split
#fighter_dataset.columns.tolist()


y=fighter_dataset['Win']
cols=['no_of_rounds',
 #'fighter',
 'age',
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
 'current_lose_streak',
 'current_win_streak',
 #'draw',
 'longest_win_streak',
 #'losses',
 'total_rounds_fought',
 'total_time_fought(seconds)',
 'total_title_bouts',
 'win_by_Decision_Majority',
 'win_by_Decision_Split',
 'win_by_Decision_Unanimous',
 'win_by_KO/TKO',
 'win_by_Submission',
 'win_by_TKO_Doctor_Stoppage',
 #'wins',
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
 'avg_opp_TOTAL_STR_landed',
 #'Win',
 'Stance_Open Stance',
 'Stance_Orthodox',
 'Stance_Sideways',
 'Stance_Southpaw',
 'Stance_Switch']
x=fighter_dataset[cols]
(x_train, x_test, y_train, y_test) = cv.train_test_split(x, y, test_size=0.2)
SEED=1


##PCA
from sklearn.decomposition import PCA
pca = PCA(.95)

pca.fit(x_train)


train_dat = pca.transform(x_train)
test_dat = pca.transform(x_test)

pca.n_components_
