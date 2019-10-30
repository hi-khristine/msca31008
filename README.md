# Project Code for Team Elbow

Data is at https://www.kaggle.com/rajeevw/ufcdata.

Some Git tips:

* The .gitignore file shows which types of files we want to avoid tracking with Git and pushing to GitHub. This includes anything in data/ (a folder for placing data in) and out/ (a folder for placing data and other output that can be generated from code and shouldn't be tracked with Git). It also includes typically-large non-text files (they slow Git down) and jupyter checkpoints.

* Git will run slow if you put data files on Github, so it is best to download data files separately instead of cloning from Git. I've added data/ to gitignore so you can place data in there and Git should ignore them.

* The README is written in [markdown](https://www.markdownguide.org/cheat-sheet/).