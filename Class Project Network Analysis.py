#!/usr/bin/env python
# coding: utf-8

# In[17]:


import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report
import sklearn.model_selection as cv
from sklearn.metrics import mean_squared_error as MSE
plt.rcParams.update({'font.size': 8})
import networkx as nx

get_ipython().run_line_magic('matplotlib', 'inline')


# In[7]:


data=pd.read_csv("data.csv")
data.head()

d=data[data['weight_class']=='Lightweight']


# In[8]:


d.head()


# In[9]:


FG = nx.from_pandas_edgelist(d, source='R_fighter', target='B_fighter', edge_attr=True,)


# In[10]:


FG.nodes()


# In[11]:


FG.edges()


# In[21]:


# your code is here (Quick view of the Graph.) 
g = nx.Graph() 
g.add_nodes_from(FG.nodes)
g.add_edges_from(FG.edges)

pos = nx.spring_layout(FG)
nx.draw_networkx_nodes(FG, pos, node_size = 30)
nx.draw_networkx_labels(FG, pos)
nx.draw_networkx_edges(FG, pos)
plt.figure(figsize=(500,500))
plt.show()
#nx.draw(g)


# In[13]:


nx.algorithms.degree_centrality(FG) # Notice the 3 airports from which all of our 100 rows of data originates

# airports = JFK, EWR, LGA
# Calculate average edge density of the Graph

# your code is here
from statistics import mean
s=pd.DataFrame.from_dict(nx.algorithms.degree_centrality(FG), orient='index')
mean(s[0])


# In[14]:


nx.average_degree_connectivity(FG)


# In[27]:


stats=nx.degree_centrality(FG)

from collections import Counter 
k=Counter(stats)
high = k.most_common(3) #most popular fighters

print(high)


# In[29]:


# Define maximal_cliques()
def maximal_cliques(G, size):
    """
    Finds all maximal cliques in graph `G` that are of size `size`.
    """
    mcs = []
    for clique in nx.find_cliques(G):
        if len(clique) == size:
            mcs.append(clique)
    return mcs

# Calculate the maximal cliques in G: cliques
cliques = nx.find_cliques(FG)

# Count and print the number of maximal cliques in G
print(len(list(cliques)))


# In[30]:


# Compute the degree centralities of G: deg_cent
deg_cent = nx.degree_centrality(FG)

# Compute the maximum degree centrality: max_dc
max_dc = max(deg_cent.values())

# Find the user(s) that have collaborated the most: prolific_collaborators
prolific_collaborators = [n for n, dc in deg_cent.items() if dc == max_dc]

# Print the most prolific collaborator(s)
print(prolific_collaborators)


# In[31]:


# Import necessary modules
from itertools import combinations
from collections import defaultdict

# Initialize the defaultdict: recommended
recommended = defaultdict(int)

# Iterate over all the nodes in G
for n, d in FG.nodes(data=True):

    # Iterate over all possible triangle relationship combinations
    for n1, n2 in combinations(FG.neighbors(n), 2):
    
        # Check whether n1 and n2 do not have an edge
        if not FG.has_edge(n1, n2):
        
            # Increment recommended
            recommended[(n1, n2)] += 1

# Identify the top 10 pairs of users
all_counts = sorted(recommended.values())
top10_pairs = [pair for pair, count in recommended.items() if count > all_counts[-10]]
print(top10_pairs)


# In[ ]:




