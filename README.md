# irboost
Fit a predictive model with the iteratively reweighted boosting (IRBoost) that minimizes the robust loss functions in the CC-family (concave-convex). The convex optimization is conducted by functional descent boosting algorithm in the R package \pkg{xgboost}. The IRBoost reduces the weight of the observation that leads to a large loss; it also provides weights to help identify outliers. Applications include the robust   generalized linear models and extensions, where the mean is related to the predictors by boosting, and robust accelerated failure time models. Wang (2021) <arXiv:2101.07718>.

How to generate the vignette document?

R CMD Sweave --pdf --clean irbst.Rnw

This requires jss.bst in the same folder.
