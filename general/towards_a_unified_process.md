# Why this document

This document serves as a guide to solving kaggle competitions and to see how we can make an unified approach to solving these problems. In theory this could then be applied to any datascience problem.



# Basic structure:

Steps:

    1. Understand the problem
        * Analyzing variables -> Some can be automated (type, #, NaN,... ), some not.
        * some numerical features could be in fact categories (for example postcodes), some categories could be indicators for numerical features
        * Suggest to automate creation easily read file to which other metadata can be easily added
        * Also write down subjective metadata (This variable probably is a good indicater)
    2. Univariable study
        * descriptive statistics
        * histograms, skewness, kurtosis
        * numericals -> scatterplots, ...
        * categorical - > boxplots, ...
        * Can easily be automated
    3. Multivariate study
        * correlation matrix ( viz: heatmap, paired scatterplots)
        * anova, tsne
        * also easily automated
    4. Basic cleaning
        * Missing data
        * duplicates
        * If to many missing, could be better not to use variable
        * if only a few occurances dont have a variable, consider deletting the occurances iso going through an effort to handle them.
        * consider outliers
        * multiple categories -> dummies.
        * skewness can often be solved by log transforming/ other transformations. (carefull with 0 values)
        * carefule to scale only on trainingset
        * Only limited automation.
    5. Test some basic assumptions
    6. Feature engineering
    7. model training
        * plot residiuals, predictions train/validation data
        * easily automated, within bounds of parameter selection
        * some model selection idea:
            * L2 probably need regularization
            * L1 regulariyation is usefull for high dimensional dataset
            * L1+L2 (elasticnet) interesting
    100. ...


Various observations:
    * Mostly same libs (pandas, numpy, scipy, scikit-learn, matplotlib, seaborn, ...)

