import os
for f in os.listdir('./fun/'): exec(open('./fun/'+f).read())
del f

import pandas as pd
import numpy as np
import math
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns

from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()

# Load data
data = pd.read_csv("./out/d_fight_level_dataset_1line.csv", index_col = 0)
data.reset_index(inplace = True)

# Change winner to binary 1/0:
data.Winner = data.Winner.apply(lambda x: np.where(x == -1, 0, 1))

# Initial features and target
features = pd.Series(data.columns, index = data.columns)
target = "Winner"

# Remove referree, date, location, winner, title_bout, weight_class, no_of_rounds
features.drop(index = ["Referee", "date", "location", "Winner", "title_bout",
                       "weight_class", "no_of_rounds"], inplace = True)

# Different sorts of wins are relatively sparce and are likely perfectly colinear
# with number of wins
features.drop(index = ["Diff_win_by_Decision_Majority",
                       "Diff_win_by_Decision_Split",
                       "Diff_win_by_Decision_Unanimous",
                       "Diff_win_by_KO/TKO",
                       "Diff_win_by_Submission",
                       "Diff_win_by_TKO_Doctor_Stoppage"], inplace = True)

# Diff_draw is mostly NA/0
features.drop(index = "Diff_draw", inplace = True)

# Delete two rows with missing data
delete = np.where(data[features].apply(lambda x: np.sum(np.isnan(x)) != 0, axis = 1))
delete = [x for x in delete for x in x]
data.drop(index = delete, inplace = True)

# Standardize columns
scaler = StandardScaler(copy = False)

X = data[features]
y = data[target]

m = RandomForestRegressor( 
    n_estimators = 400, 
    min_samples_leaf = 0.05, 
    random_state = 841)

# Expanding window
dates = np.unique(data.date)
split = math.floor(len(dates) * .6)

len(data[data.date <= dates[split]].index)

maxf = len(data[data.date <= dates[split]].index)

X_train = data[features].iloc[range(maxf)]
y_train = data[target].iloc[range(maxf)]

scaler.fit(X_train)
X_train = scaler.transform(X_train, copy = False)

m.fit(X_train, y_train)

y_pred = pd.DataFrame(data = {"prediction" : m.predict(X_train)})
y_pred["actual"] = y_train.reset_index(drop = True)
y_pred["pred_bucket"] = pd.cut(y_pred.prediction, bins = np.arange(.35, 1, .05))

y_diagnostics = y_pred[["actual", "pred_bucket"]].groupby("pred_bucket").aggregate(["sum", "count"])
y_diagnostics.columns = ["sum", "count"]
y_diagnostics["pct"] = y_diagnostics["sum"] / y_diagnostics["count"]

def determine_result(x):
    if x.action == "Bet Win":
        if x.actual == 1:
            return "Won"
        else:
            return "Lost"
    elif x.action == "Bet Loss":
        if x.actual == 0:
            return "Won"
        else:
            return "Lost"
    else:
        return "No Action"
    
full_results = pd.DataFrame()

for x in range(split, len(dates)):
    maxf = len(data[data.date <= dates[x]].index)
    X_train = data[features].iloc[range(maxf)]
    scaler.fit(X_train)
    X_train = scaler.transform(X_train, copy = False)
    y_train = data[target].iloc[range(maxf)]
    m.fit(X_train, y_train)
    
    fights = np.where(data.date == dates[x])
    X_test = data[features].iloc[fights]
    X_test = scaler.transform(X_test, copy = False)
    y_test = data[target].iloc[fights]
    
    result = pd.DataFrame(data = {"date" : data.iloc[fights].date,
                                  "prediction" : m.predict(X_test),
                                  "actual" : y_test})
    result["action"] = result.apply(lambda x: np.where(x.prediction < .55, "Bet Loss",
          np.where(x.prediction > .65, "Bet Win", "No Action")), axis = 1)
    result["result"] = result.apply(determine_result, axis = 1)
    result["wager_result"] = result.apply(lambda x: np.where(x.result == "Won", 100,
    np.where(x.result == "Lost", -100, 0)), axis = 1)
    full_results = full_results.append(result, ignore_index = True)

full_results["cumulative_wager_result"] = full_results.wager_result.cumsum()

full_results_summary = full_results[["date", "cumulative_wager_result"]].groupby("date").aggregate(np.max)
full_results_summary["date"] = full_results_summary.index
full_results_summary.date = full_results_summary.date.apply(lambda x: datetime.datetime.strptime(x, "%Y-%m-%d"))

sns.set(font_scale = .75)
sns.lineplot(x = "date", y = "cumulative_wager_result", data = full_results_summary)
