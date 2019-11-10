import pandas as pd

data = pd.read_csv('./data/data.csv')

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


## Join the datasets
fighter_dataset = red_fighters.append(blue_fighters)

fighter_dataset.sort_values(by = ['fighter', 'date'], inplace = True)

import os
os.getcwd()
fighter_dataset.to_csv('./data/fighter_level_dataset.csv')
