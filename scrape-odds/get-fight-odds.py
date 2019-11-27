initurl = 'https://www.bestfightodds.com/archive'

from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import time

# read in functions.
import os
for f in os.listdir('../fun/'): exec(open('../fun/'+f).read())
del f

driver = webdriver.Chrome( "chromedriver.exe" )
driver.get( initurl )

if 'get-fight-odds.pkl' not in os.listdir('.'):
    f = pd.read_csv( '../out/b-handle-nas.csv' )[[ 'fightid', 'date', 'location', 'R_fighter', 'B_fighter' ]]
    f = f.assign( R_odds = np.nan, B_odds = np.nan )
    f.date = pd.to_datetime( f.date )
    f['R_lastname'] = f.R_fighter.str.replace( '^.+ ', '' ).str.lower()
    f['B_lastname'] = f.B_fighter.str.replace( '^.+ ', '' ).str.lower()
    for i in [ 'R_fighter', 'B_fighter' ]:
        f[i+'_match'] = f[i].str.lower().str.replace('[^a-z]','')
        del i
    searchedfighters = []
    fighters = np.unique( pd.concat( [ f.R_fighter[ f.R_odds.isna() ], f.B_fighter[ f.B_odds.isna()] ] ) )    
else:
    load( 'get-fight-odds.pkl' )
    #f = pd.read_csv('../out/get-fight-odds.csv')
    #f.date = pd.to_datetime(f.date)

f[ f.isna().any( axis = 1 ) ].head()

matchon = 'lastname'
#matchon = 'fighter_match'

while len(fighters) > 0 and ( f[['R_odds','B_odds']].isna().sum().sum() > 0 ):    
    
    fighter = fighters[0]
    print( "Searching: " + fighter )
    found = True
    
    search = driver.find_element_by_name("query")
    search.send_keys( fighter )
    search.send_keys( Keys.RETURN )
    del search 
        
    try:
        
        if len( driver.find_elements_by_css_selector( 'th.oppcell' ) ) == 0:
            
            foundresult = driver.find_elements_by_xpath("//a[text()='" + fighter +"']")
            if len(foundresult) > 1: 
                print( 'Multiple matches found for: ' + fighter )
                found = False
            if len(foundresult) == 0: 
                print( 'Not matches found for: ' + fighter )
                found = False
            else:
                foundresult [0].click()
            del foundresult 
        
        if found:
            
                fighters = [ x.text for x in driver.find_elements_by_css_selector( 'th.oppcell' ) ]
                odds = [ x.text for x in driver.find_elements_by_css_selector( 'td.moneyline' ) ]
                events = [ x.text for x in driver.find_elements_by_css_selector( 'td.item-non-mobile' ) ]
                
                dt = None
                i = 0
                while len(events) > 0:
                    if events[0] != 'Future Events':
                        row = pd.DataFrame({
                            'date': events[1],
                            'R_fighter': fighters[0], 'B_fighter': fighters[1],
                            'R_odds': odds[0], 'B_odds': odds[3]
                        }, index = [i] )
                        dt = pd.concat([dt,row])
                        i = i + 1
                    fighters = fighters[2:]
                    events = events[2:]
                    odds = odds[6:]
                    
                # now fill in the odds in the data.
                
                dt.date = pd.to_datetime(dt.date)
                
                for i in [ 'R_fighter', 'B_fighter' ]:
                    dt[i+'_match'] = dt[i].str.lower().str.replace('[^a-z]','')
                    del i
                    
                for index, row in dt.iterrows():
                    if row.R_odds == '': continue
                    match = ( f.date == row.date ) & ( f[ 'R_'+ matchon ] == row[ 'R_'+ matchon ] ) & ( f[ 'B_'+ matchon ] == row[ 'B_'+ matchon ]  )
                    if match.any():
                        f.loc[ match & ( f.R_odds.isna() ), 'R_odds' ] = row.R_odds
                        f.loc[ match & ( f.B_odds.isna() ), 'B_odds' ] = row.B_odds
                        #print( "Filled: %s" % match.sum() )
                    else:
                        match = ( f.date == row.date ) & ( f[ 'R_'+ matchon ] == row[ 'b_'+ matchon ] ) & ( f[ 'B_'+ matchon ] == row[ 'R_'+ matchon ]  )
                        if match.any():
                            f.loc[ match & ( f.R_odds.isna() ), 'R_odds' ] = row.B_odds
                            f.loc[ match & ( f.B_odds.isna() ), 'B_odds' ] = row.R_odds
                            #print( "Filled: %s" % match.sum() )
                    del match, index, row
                    
                    # code for testing matches.
                    #dt.sort_values( 'date' )
                    #f[ ( f.R_fighter_match == 'aaronrosa' ) | ( f.B_fighter_match == 'aaronrosa' ) ]
                    
    except:
        pass
            
    # now prep for the next loop.
    searchedfighters.append( fighter )
    fighters = setdiff(
        np.unique( pd.concat( [ f.R_fighter[ f.R_odds.isna() ], f.B_fighter[ f.B_odds.isna()] ] ) ),
        searchedfighters
    )
    save( 'get-fight-odds.pkl', f, searchedfighters, fighters )
    print( "Fighters left: %s \t NAs left: %s" % ( len(fighters), f[['R_odds','B_odds']].isna().sum().sum() ) )
    time.sleep(4)
    #  1408 8196
    
