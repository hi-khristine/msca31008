
These files should be run in order to create the various datasets used by models and exploratory data analysis.



Built data files should be saved in the out/ folder to clarify that these are created, not raw, files.



Files before `c-` prefix are fight-level. Files with `c-` prefix are fighter-level.



I use a mix of `save` which allows saving multiple objects to a `.pkl` file for reading in at the next step, and regular csv files when only one object needs to pass to the next step.



Please add to this pipeline in such a way that the files can be run in order without errors.