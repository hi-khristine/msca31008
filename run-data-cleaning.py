# author: Bryce C
# purpose: run all scripts in data-cleaning.
# note: 
#   this code should be run from the msca31008 directory.
#   when it finishes, processed files will be saved in the out/ directory.

# this file errors out and I'm not sure why.

# read in functions.
import os, sys, fnmatch, traceback

os.chdir( './data-cleaning' )

for datacleanfile in fnmatch.filter( os.listdir('.'), '*.py' ): 
    print(datacleanfile)
    try: 
        #exec( open(datacleanfile).read() )
        runfile(datacleanfile, wdir = os.getcwd())
    except:
        #einfo = sys.exc_info()
        #traceback.print_last( einfo )
        raise Exception( "Error at file [ " + datacleanfile + " ]." )
    del datacleanfile

