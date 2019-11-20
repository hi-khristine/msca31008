import pandas as pd
import numpy as np
import os, sys

os.chdir(r"C:\Users\Adam-PC\Documents\GitHub\msca31008")

for f in os.listdir('fun/'): exec(open('fun/'+f).read())
#load(r"C:\Users\Adam-PC\Documents\GitHub\msca31008\out\c2-fighter-level-fillna.pkl")

data = pd.read_csv(r"C:\Users\Adam-PC\Documents\GitHub\msca31008\data\fighter_level_dataset_with_clusters.csv", index_col = 0)
data.drop(columns = ['Unnamed: 0.1', 'Winner'], inplace = True)

from sklearn.neighbors import KernelDensity
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

names = data['fighter']
y = data['kmeans_label']
X = data.drop(columns = ['fighter', 'kmeans_label'])
X = pd.get_dummies(X)


pca = PCA(copy = True).fit(X)

###################################
##### Find the PCA Parameters #####
###################################

lmts = [0.75, 0.9, 0.95, 0.99]
vars = np.cumsum(pca.explained_variance_ratio_)
vls = np.array([np.argmax(vars > i) for i in lmts])

# Make the plot
sns.set(style="ticks", font_scale=2.0)
fig, ax = plt.subplots(figsize=(10,6))

# Decorate the plot
ax.set_xlabel('Dimension #')
ax.set_ylabel('Explained Variance Ratio')
ax.set_title('Fraction of Explained Variance')

ax.set_xlim(0, 10.0)
ax.set_ylim(0, 0.15)

# Draw lines for the cumulative variance
ax.vlines(vls, 0.0, 0.05, linestyles='dashed') #colors=sns.xkcd_rgb)
for xl, txt in zip(vls, lmts):
    ax.text(xl, 0.055, str(txt), fontsize = 18, \
            color=sns.xkcd_rgb["pale red"], \
            horizontalalignment='center')
    
# Now draw the points, with bolder colors.
plt.plot(pca.explained_variance_ratio_, color=sns.xkcd_rgb["denim blue"], linewidth=3)

plt.show()
sns.despine(offset=5, trim=True)

#################################################
##### Kernel Estimation for Data Generation #####
#################################################

pca = PCA(n_components = 8, copy = True)
pca_data = pca.fit_transform(X)
kde = KernelDensity(kernel = 'gaussian')
kde.fit(pca_data)

new_data = kde.sample(500, random_state = 0)
new_data = pca.inverse_transform(new_data)
print(new_data.shape)
