# author: Bryce C working off code from Khristine.
# purpose: add dummy variables to replace categorical.
# note: 
#   this code should be run from the data-cleaning directory.
#   when it runs, you'll have a corrected CSV file in the out/ folder
#   called c3-figher-level-dummies.csv

# read in functions.
import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

load( '../out/c3-fighter-level-dummies.pkl'  )

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
 #'Stance_Sideways',
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
