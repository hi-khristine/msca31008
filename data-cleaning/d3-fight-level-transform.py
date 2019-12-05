import pandas as pd
import numpy as np
import re
import copy

# read in functions.
import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

# Load data
load( '../out/d2-fight-level-standardize-normalize-kmeansNA.pkl' )

# Make column name format consistent
cols = pd.Series(cols)
data = pd.DataFrame(X)
X = None
del X
data.columns = cols
data.rename(columns = cols.apply(lambda x: re.sub("_ATT", "_att", x)))
savey = y

# Function to calculate percentages, assigning zero where there are no attempts
# or missing data
def pct_nozero(x):
    if np.isnan(x[0]) or np.isnan(x[1]) or x[1] == 0:
        return 0
    else:
        return x[0] / x[1]

# Create percentages out of landed and attempts columns
attacks = ["BODY", "CLINCH", "DISTANCE", "GROUND", "HEAD", "LEG", "SIG_STR", "TD", "TOTAL_STR"]
colors = ["R", "B"]

R_pct_cols_fighter = list()
B_pct_cols_fighter = list()
R_pct_cols_opp = list()
B_pct_cols_opp = list()
for x in colors:
    for y in attacks:
        data[x + "_avg_" + y + "_pct"] = data[[x + "_avg_" + y + "_landed", 
            x + "_avg_" + y + "_att"]].apply(lambda z: pct_nozero(z), axis = 1)
        data[x + "_avg_opp_" + y + "_pct"] = data[[x + "_avg_" + y + "_landed",
            x + "_avg_opp_" + y + "_att"]].apply(lambda z: pct_nozero(z), axis = 1)
        data.drop(columns = [x + "_avg_" + y + "_landed",
                             x + "_avg_" + y + "_att",
                             x + "_avg_opp_" + y + "_landed",
                             x + "_avg_opp_" + y + "_att"], inplace = True)
        if x == "R":
            R_pct_cols_fighter.append(x + "_avg_" + y + "_pct")
            R_pct_cols_opp.append(x + "_avg_opp_" + y + "_pct")
        else:
            B_pct_cols_fighter.append(x + "_avg_" + y + "_pct")
            B_pct_cols_opp.append(x + "_avg_opp_" + y + "_pct")

doods = ( 'iterationnum' not in globals() ) or ( iterationnum >= 2 )

red_fighter_columns = ['R_age', 'R_Height_cms', 'R_Reach_cms', 'R_Weight_lbs']
red_stats_columns = ['R_avg_KD', 'R_avg_PASS', 'R_avg_REV', 'R_avg_SUB_ATT', 'R_avg_opp_SUB_ATT' ] + R_pct_cols_fighter + ( ['R_odds'] if doods else [] )
red_history_columns = ['R_current_lose_streak', 'R_current_win_streak', 'R_draw', 'R_longest_win_streak', 
                       'R_losses', 'R_total_rounds_fought', 'R_total_time_fought(seconds)', 'R_total_title_bouts', 
                       'R_win_by_Decision_Majority', 'R_win_by_Decision_Split', 'R_win_by_Decision_Unanimous', 
                       'R_win_by_KO/TKO', 'R_win_by_Submission', 'R_win_by_TKO_Doctor_Stoppage', 'R_wins']
red_opp_stats_columns = ['R_avg_opp_KD', 'R_avg_opp_PASS', 'R_avg_opp_REV'] +R_pct_cols_opp


blue_fighter_columns = ['B_age', 'B_Height_cms', 'B_Reach_cms', 'B_Weight_lbs', 'B_avg_SUB_ATT', 'B_avg_opp_SUB_ATT' ]
blue_stats_columns = ['B_avg_KD', 'B_avg_PASS', 'B_avg_REV'] + B_pct_cols_fighter + ( ['B_odds'] if doods else [] )
blue_history_columns= ['B_current_lose_streak', 'B_current_win_streak', 'B_draw', 'B_longest_win_streak', 
                       'B_losses', 'B_total_rounds_fought', 'B_total_time_fought(seconds)', 'B_total_title_bouts', 
                       'B_win_by_Decision_Majority', 'B_win_by_Decision_Split', 'B_win_by_Decision_Unanimous', 
                       'B_win_by_KO/TKO', 'B_win_by_Submission', 'B_win_by_TKO_Doctor_Stoppage', 'B_wins']
blue_opp_stats_columns = ['B_avg_opp_KD', 'B_avg_opp_PASS', 'B_avg_opp_REV'] + B_pct_cols_opp

fight_columns = setdiff( 
    data.columns, 
    red_fighter_columns + red_stats_columns + red_history_columns + red_opp_stats_columns + blue_fighter_columns +  \
        blue_opp_stats_columns + blue_stats_columns + blue_history_columns + blue_history_columns
)

## Organize data from the perspective of red fighters, then blue fighters, then combine.
## This produces two rows for each fight.
red_fighters = copy.deepcopy(data[ fight_columns + red_fighter_columns + red_stats_columns + 
                                  red_history_columns + red_opp_stats_columns + blue_fighter_columns + 
                                  blue_stats_columns + blue_history_columns + blue_opp_stats_columns])
red_fighters.columns = pd.Series(red_fighters.columns).apply(lambda x: re.sub("^R_", "Fighter_", x))
red_fighters.columns = pd.Series(red_fighters.columns).apply(lambda x: re.sub("^B_", "Opponent_", x))


blue_fighters = copy.deepcopy(data[ fight_columns + blue_fighter_columns + blue_stats_columns + 
                                   blue_history_columns + blue_opp_stats_columns + red_fighter_columns + 
                                   red_stats_columns + red_history_columns + red_opp_stats_columns])
blue_fighters.columns = pd.Series(blue_fighters.columns).apply(lambda x: re.sub("^B_", "Fighter_", x))
blue_fighters.columns = pd.Series(blue_fighters.columns).apply(lambda x: re.sub("^R_", "Opponent_", x))

fight_dataset = red_fighters.append(blue_fighters)

## Save dataset
fight_dataset.to_csv('../out/d_fight_level_dataset_2lines.csv')

## Another take: differences between the fighters
def remove_R(cols):
    cols = pd.Series(cols).apply(lambda x: re.sub("^R_", "_", x))
    return cols

fighter_columns = remove_R(red_fighter_columns[1:-1])
stats_columns = remove_R(red_stats_columns)
history_columns = remove_R(red_history_columns)
opp_stats_columns = remove_R(red_opp_stats_columns)

cols = fighter_columns.append(stats_columns).append(history_columns).append(opp_stats_columns)

fight_dataset = copy.deepcopy(data[fight_columns])
for x in cols:
    fight_dataset["Mean" + x] = ( data["R" + x] + data["B" + x] ) / 2
    fight_dataset["Diff" + x] = ( data["R" + x] - data["B" + x] )
    #fight_dataset["Diffovermean" + x] = fight_dataset["Diff" + x] / fight_dataset["Mean" + x]
    #fight_dataset["Diffovermean" + x][ fight_dataset["Mean" + x] == 0 ] = 0
    
# drop singular columns.
fight_dataset.drop( [ x for x in fight_dataset.columns if fight_dataset[x].value_counts().shape[0] == 1 ], axis = 1, inplace = True )

X = fight_dataset.reset_index( drop = True )
y = savey.reset_index( drop = True )
winmults = winmults.reset_index( drop = True )

# standardize/normalize again.
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
cols = X.columns
X = StandardScaler().fit_transform(X)
X = Normalizer().fit_transform(X)

save( '../out/d3-fight-level-transform.pkl', X, y, cols, winmults )

del fighter_columns, stats_columns, history_columns, opp_stats_columns, B_pct_cols_fighter, \
    B_pct_cols_opp, R_pct_cols_fighter, R_pct_cols_opp, attacks, blue_fighters, blue_history_columns, \
    blue_fighter_columns, blue_stats_columns, colors, cols, data, fight_columns, \
    red_fighters, red_history_columns, red_fighter_columns, red_stats_columns, blue_opp_stats_columns, \
    red_opp_stats_columns, x, y, fight_dataset, savey, doods
    
del X