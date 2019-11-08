import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

data = pd.read_csv(r'C:\Users\Adam-PC\Desktop\Homework\MSCA 31008 - Data Mining\Project\fighter_level_dataset.csv')

## Select the relevant data
fighter_columns = ['fighter', 'age', 'Height_cms', 'Reach_cms', 'Weight_lbs', 'Stance', 'date']

fighter_history_columns = history_columns= ['current_lose_streak', 'current_win_streak', 'draw', 'longest_win_streak', 'losses', 'total_rounds_fought', 'total_time_fought(seconds)', 'total_title_bouts', 'win_by_Decision_Majority', 'win_by_Decision_Split', 'win_by_Decision_Unanimous', 'win_by_KO/TKO', 'win_by_Submission', 'win_by_TKO_Doctor_Stoppage', 'wins']

relevant_data = data[fighter_columns + fighter_history_columns]

## Remove duplicates of fighters, only keep the most up to date entry
relevant_data.sort_values(by = 'date', ascending = False, inplace = True)
relevant_data = relevant_data.drop_duplicates(subset = 'fighter')
relevant_data.drop(columns = 'date', inplace = True)
relevant_data.drop(columns = 'Stance', inplace = True) # Removed Stance for dataset simplicity, consider adding in later
relevant_data.fillna(0, inplace = True)

x = relevant_data.iloc[:, 1:]
y = relevant_data.iloc[:, 0]

model = TSNE(learning_rate = 100)

tsne_transformed = model.fit_transform(x)

xs = tsne_transformed[:, 0]
ys = tsne_transformed[:, 1]

## Plot the clusters
plt.scatter(xs, ys, c = y)
