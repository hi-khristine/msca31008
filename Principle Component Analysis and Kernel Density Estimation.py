import pandas as pd
import numpy as np
from sklearn.neighbors import KernelDensity
import os, sys

os.chdir(r"C:\Users\Adam-PC\Documents\GitHub\msca31008")


for f in os.listdir('fun/'): exec(open('fun/'+f).read())
load(r"C:\Users\Adam-PC\Documents\GitHub\msca31008\out\c2-fighter-level-fillna.pkl")


fighter_dataset.columns
fighter_dataset['fighter'].value_counts().hist()
fighter_dataset['fighter'].value_counts().mean()


fighter_columns = ['fighter', 'age', 'Height_cms', 'Reach_cms', 'Weight_lbs', 'Stance', 'date']
fighter_history_columns = ['current_lose_streak', 'current_win_streak', 'draw', 'longest_win_streak', 'losses', 'total_rounds_fought', 'total_time_fought(seconds)', 'total_title_bouts', 'win_by_Decision_Majority', 'win_by_Decision_Split', 'win_by_Decision_Unanimous', 'win_by_KO/TKO', 'win_by_Submission', 'win_by_TKO_Doctor_Stoppage', 'wins']
fighter_style_columns = ['avg_BODY_att', 'avg_CLINCH_att', 'avg_DISTANCE_att', 'avg_GROUND_att', 'avg_HEAD_att', 'avg_KD', 'avg_LEG_att', 'avg_PASS', 'avg_REV', 'avg_SIG_STR_att', 'avg_SUB_ATT', 'avg_TD_att', 'avg_TOTAL_STR_att']

relevant_data = fighter_dataset[fighter_columns + fighter_history_columns + fighter_style_columns]



## Remove duplicates of fighters, only keep the most up to date entry
relevant_data.sort_values(by = 'date', ascending = False, inplace = True)
relevant_data = relevant_data.drop_duplicates(subset = 'fighter')
relevant_data.drop(columns = 'date', inplace = True)

## Separate the fighters from the stats
y = relevant_data['fighter']
X = relevant_data.drop('fighter', axis = 1)
X = pd.get_dummies(X)

###################
##### K Means #####
###################

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

## Scale the data so features with inherently large values don't overshadow others
#stscaler = StandardScaler().fit(X)
#scaled_data = stscaler.transform(X)

wcss = []


# Using the elbow method to identify optimal number of clusters > find when the data plateaus
for i in range(1, 15):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',
                    max_iter = 400, n_init = 10, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1, 15), wcss)

# Creating the model with the optimal number of clusters
kmeans = KMeans(n_clusters = 10, init = 'k-means++', max_iter = 500, n_init = 10, random_state = 0)

# Fit the data and predict values
kmeans_labels = kmeans.fit_predict(X) # This creates an array with the same number of observations as x.  Each is a value corresonding to the cluster the point was assigned to.

# Find the calculated centers
kmeans.cluster_centers_
kmeans.cluster_centers_[:, 0]

# Visualizing the data
relevant_data['kmeans_label'] = kmeans_labels
clustered_fighters = relevant_data[['fighter', 'kmeans_label']]

# Groupings
clusters_grouped = relevant_data.drop('Stance', axis = 1).groupby('kmeans_label').agg('mean')

plt.bar(clusters_grouped.columns, clusters_grouped.loc[0])
plt.bar(clusters_grouped.columns, clusters_grouped.loc[1])
plt.bar(clusters_grouped.columns, clusters_grouped.loc[2])
plt.bar(clusters_grouped.columns, clusters_grouped.loc[3])

# x[y_kmeans == 0, 0] is the 0th column of x where the predicted category (of y_kmeans) is 0
#plt.scatter(X[kmeans_labels == 0, 0], X[kmeans_labels == 0, 1], s = 75, c = 'red', label = 'category 0')
#plt.scatter(X[kmeans_labels == 1, 0], X[kmeans_labels == 1, 1], s = 75, c = 'blue', label = 'category 1')
#plt.scatter(X[kmeans_labels == 2, 0], X[kmeans_labels == 2, 1], s = 75, c = 'green', label = 'category 2')
#plt.scatter(X[kmeans_labels == 3, 0], X[kmeans_labels == 3, 1], s = 75, c = 'yellow', label = 'category 3')
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'black', label = 'Centroide')
#plt.legend()


X[kmeans_labels == 0]
