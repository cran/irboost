# irboost
Fit a predictive model using the Iteratively Reweighted Boosting   (IRBoost) to minimize robust loss functions within the CC-family (concave-      convex). This constitutes an application of Iteratively Reweighted Convex       Optimization (IRCO), where convex optimization is performed using the           functional descent boosting algorithm. IRBoost assigns weights to facilitate    outlier identification. Applications include robust generalized linear models   and robust accelerated failure time models.

How to generate the vignette document?

R CMD Sweave --pdf --clean irbst.Rnw

This requires jss.bst in the same folder.
