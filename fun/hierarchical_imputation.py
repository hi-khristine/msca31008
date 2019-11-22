# author: Michael C
# impute at fighter level, then weight class level
 
# function to impute columns w/ NAs
def impute_with_mean(df):

    """iterate through columns of pd df and replace nulls with mean"""      
                          
    # loop through columns
    for column in fight_stats_cols:

        # transfer column to separate series
        col_data = df[column]

        # check for missing values
        missing_data = col_data.isna().sum()

        if missing_data > 0:
            # replace missing data with mean at fighter level of hierarchy then at weight class level if 
            # fighter level is not the solve

            df.column = df.groupby('fighter')[column].apply(lambda x: x.fillna(x.mean()))
            df.column = df.groupby('wt_class')[column].apply(lambda x: x.fillna(x.mean()))

            df[column] = df.column

    return df