#' fit a robust predictive model with iteratively reweighted boosting algorithm
#'
#' Fit a predictive model with the iteratively reweighted convex optimization (IRCO) that minimizes the robust loss functions in the CC-family (concave-convex). The convex optimization is conducted by functional descent boosting algorithm in the R package \pkg{xgboost}. The iteratively reweighted boosting (IRBoost) algorithm reduces the weight of the observation that leads to a large loss; it also provides weights to help identify outliers. Applications include the robust generalized
#' linear models and extensions, where the mean is related to the predictors by boosting, and robust accelerated failure time models. 
#' @param data input data, if \code{objective="survival:aft"}, it must be an \code{xgb.DMatrix}; otherwise, it can be a matrix of dimension nobs x nvars; each row is an observation vector. Can accept \code{dgCMatrix}
#' @param label response variable. Quantitative for \code{objective="reg:squarederror"},\cr 
#'        \code{objective="count:poisson"} (non-negative counts) or \code{objective="reg:gamma"} (positive). 
#'        For \code{objective="binary:logitraw" or "binary:hinge"}, \code{label} should be a factor with two levels
#' @param weights vector of nobs with non-negative weights 
#' @param params the list of parameters, \code{params} is passed to function xgboost::xgboost which requires the same argument. The list must include \code{objective}, a convex component in the CC-family, the second C, or convex down. It is the same as \code{objective} in the \code{xgboost::xgboost}. The following objective functions are currently implemented: 
#'   \itemize{
#'     \item \code{reg:squarederror} Regression with squared loss.
#'     \item \code{binary:logitraw} logistic regression for binary classification, predict linear predictor, not probabilies.
#'     \item \code{binary:hinge} hinge loss for binary classification. This makes predictions of -1 or 1, rather than   producing probabilities.
#'     \item \code{multi:softprob} softmax loss function for multiclass problems. The result contains predicted probabilities of each data point in each class, say p_k, k=0, ..., nclass-1. Note, \code{label} is coded as in [0, ..., nclass-1]. The loss function cross-entropy for the i-th observation is computed as -log(p_k) with k=lable_i, i=1, ..., n.
#'     \item \code{count:poisson}: Poisson regression for count data, predict mean of poisson distribution.
#'     \item \code{reg:gamma}: gamma regression with log-link, predict mean of gamma distribution. The implementation in \code{xgboost} takes a parameterization in the exponential family:\cr
#' xgboost/src/src/metric/elementwise_metric.cu.\cr
#' In particularly, there is only one parameter psi and set to 1. The implementation of the IRCO algorithm follows this parameterization. See Table 2.1, McCullagh and Nelder, Generalized linear models, Chapman & Hall, 1989, second edition.
#'     \item \code{reg:tweedie}: Tweedie regression with log-link. See also\cr \code{tweedie_variance_power} in range: (1,2). A value close to 2 is like a gamma distribution. A value close to 1 is like a Poisson distribution.
#'     \item \code{survival:aft}: Accelerated failure time model for censored survival time data. \code{irboost} invokes \code{irb.train_aft}. 
#'}
#' @param z_init vector of nobs with initial convex component values, must be non-negative with default values = weights if provided, otherwise z_init = vector of 1s 
#' @param cfun concave component of CC-family, can be \code{"hacve", "acave", "bcave", "ccave"}, 
#' \code{"dcave", "ecave", "gcave", "hcave"}.\cr
#' See Table 2 at https://arxiv.org/pdf/2010.02848.pdf
#' @param s tuning parameter of \code{cfun}. \code{s > 0} and can be equal to 0 for \code{cfun="tcave"}. If \code{s} is too close to 0 for    \code{cfun="acave", "bcave", "ccave"}, the calculated weights can become 0 for all observations, thus crash the program
#' @param delta a small positive number provided by user only if \code{cfun="gcave"} and \code{0 < s <1}
#' @param iter number of iteration in the IRCO algorithm
#' @param nrounds boosting iterations within each IRCO iteration
#' @param del convergency criteria in the IRCO algorithm, no relation to \code{delta}
#' @param trace if \code{TRUE}, fitting progress is reported
#' @param ... other arguments passing to \code{xgboost} 
#' @importFrom stats predict
#' @importFrom xgboost xgboost
#' @importFrom mpath compute_wt compute_g loss3 y2num y2num4glm
#' @return An object with S3 class \code{xgboost} with the additional elments:
#' \itemize{
#'   \item \code{weight_update_log} a matrix of \code{nobs} row by \code{iter}      column of observation weights in each iteration of the IRCO algorithm
 #'   \item \code{weight_update} a vector of observation weights in the last IRCO iteration that produces the final model fit
#'   \item\code{loss_log} sum of loss value of the composite function in each IRCO iteration. Note, \code{cfun} requires \code{objective} non-negative in some cases. Thus care must be taken. For instance, with \code{objective="reg:gamma"}, the loss value is defined by gamma-nloglik - (1+log(min(y))), where y=label. The second term is introduced such that the loss value is non-negative. In fact, gamma-nloglik=y/ypre + log(ypre) in the \code{xgboost}, where ypre is the mean prediction value, can
#'   be negative. It can be derived that for fixed \code{y}, the minimum value of gamma-nloglik is achived at ypre=y, or 1+log(y). Thus, among all \code{label} values, the minimum of gamma-nloglik is 1+log(min(y)).
#'}
#' @author Zhu Wang\cr Maintainer: Zhu Wang \email{zhuwang@gmail.com}
#' @references Wang, Zhu (2024), \emph{Unified Robust Boosting}, Zhu Wang, Unified Robust Boosting, Journal of Data Science (2024), 1-19, DOI 10.6339/24-JDS1138
#' @keywords regression classification
#' @export irboost
#' @examples
#' \donttest{
#' # regression, logistic regression, Poisson regression
#' x <- matrix(rnorm(100*2),100,2)
#' g2 <- sample(c(0,1),100,replace=TRUE)
#' fit1 <- irboost(data=x, label=g2, cfun="acave",s=0.5, 
#'                 params=list(objective="reg:squarederror", max_depth=1), trace=TRUE, 
#'                 verbose=0, nrounds=50)
#' fit2 <- irboost(data=x, label=g2, cfun="acave",s=0.5, 
#'                 params=list(objective="binary:logitraw", max_depth=1), trace=TRUE,  
#'                 verbose=0, nrounds=50)
#' fit3 <- irboost(data=x, label=g2, cfun="acave",s=0.5, 
#'                 params=list(objective="binary:hinge", max_depth=1), trace=TRUE,  
#'                 verbose=0, nrounds=50)
#' fit4 <- irboost(data=x, label=g2, cfun="acave",s=0.5, 
#'                 params=list(objective="count:poisson", max_depth=1), trace=TRUE,      
#'                 verbose=0, nrounds=50)
#'
#' # Gamma regression
#' x <- matrix(rnorm(100*2),100,2)
#' g2 <- sample(rgamma(100, 1))
#' library("xgboost")
#' param <- list(objective="reg:gamma", max_depth=1)
#' fit5 <- xgboost(data=x, label=g2, params=param, nrounds=50)
#' fit6 <- irboost(data=x, label=g2, cfun="acave",s=5, params=param, trace=TRUE, 
#'                 verbose=0, nrounds=50)
#' plot(predict(fit5, newdata=x), predict(fit6, newdata=x))
#' hist(fit6$weight_update)
#' plot(fit6$loss_log)
#' summary(fit6$weight_update)
#' 
#' # Tweedie regression 
#' param <- list(objective="reg:tweedie", max_depth=1)
#' fit6t <- irboost(data=x, label=g2, cfun="acave",s=5, params=param, 
#'                  trace=TRUE, verbose=0, nrounds=50)
#' # Gamma vs Tweedie regression
#' hist(fit6$weight_update)
#' hist(fit6t$weight_update)
#' plot(predict(fit6, newdata=x), predict(fit6t, newdata=x))
#'
#' # multiclass classification in iris dataset:
#' lb <- as.numeric(iris$Species)-1
#' num_class <- 3
#' set.seed(11)
#' 
#' param <- list(objective="multi:softprob", max_depth=4, eta=0.5, nthread=2, 
#' subsample=0.5, num_class=num_class)
#' fit7 <- irboost(data=as.matrix(iris[, -5]), label=lb, cfun="acave", s=50,
#'                 params=param, trace=TRUE, verbose=0, nrounds=10)
#' # predict for softmax returns num_class probability numbers per case:
#' pred7 <- predict(fit7, newdata=as.matrix(iris[, -5]))
#' # reshape it to a num_class-columns matrix
#' pred7 <- matrix(pred7, ncol=num_class, byrow=TRUE)
#' # convert the probabilities to softmax labels
#' pred7_labels <- max.col(pred7) - 1
#' # classification error: 0!
#' sum(pred7_labels != lb)/length(lb)
#' table(lb, pred7_labels)
#' hist(fit7$weight_update)
#'
#' }
irboost <- function(data, label, weights, params = list(), z_init=NULL, cfun="ccave", s=1, delta=0.1, iter=10, nrounds=100, del=1e-10, trace=FALSE, ...){
  call <- match.call()
  dfun <- params$objective
  if(dfun=="survival:aft"){
  return(irb.train_aft(params=params, data=data, cfun=cfun, s=s, delta=delta, iter=iter, nrounds=nrounds, del=del, trace=trace, ...))
  }
  if(!dfun %in% c("reg:squarederror", "binary:logitraw", "binary:hinge", "multi:softprob", "count:poisson", "reg:gamma", "reg:tweedie"))
    stop("dfun not implemented/applicable")
  x <- data
  y <- label
  if(dfun %in% c("reg:gamma") && any(y <= 0))
    stop("response variable y must be positive for dfun ", dfun)
  if(dfun %in% c("binary:logitraw", "binary:hinge")){
    ynew <- eval(parse(text="mpath::y2num(y)"))
    y <- eval(parse(text="mpath::y2num4glm(y)"))
  }else 
    ynew <- y
  if(!is.null(z_init)){
      if(length(z_init)!=length(y))
        stop("z_init must be the same length of response variable y") 
      if(any(z_init < 0))
        stop("z_init must be non-negative")
  }
  cfunval <- eval(parse(text="mpath::cfun2num(cfun)"))
  #what if dfun is not defined, such as gamma? it is worth to updating mpath
  dfunval <- switch(dfun,
                    "reg:squarederror"=1,
                    "binary:logitraw"=5,
                    "binary:hinge"=6,
                    "count:poisson"=8,
                    "reg:gamma"=NULL,
                    "reg:tweedie"=NULL,
                    "multi:softprob"=NULL)
  eval(parse(text="mpath::check_s(cfun, ifelse(is.null(dfunval), 1, dfunval), s)"))
  d <- 10 
  k <- 1
  if(trace) {
    cat("\nrobust boosting ...\n")
  }
  loss_log <- rep(NA, iter)
  n <- length(y)
  weight_update_log <- matrix(NA, nrow=n, ncol=iter)
  if(missing(weights)) weights <- rep(1, n)
  if(is.null(z_init)) ylos <- weights else ylos <- z_init #initial values
  if(dfun=="reg:gamma")
    min_nloglik <- 1+log(min(y)) #the minimum value of negative log-likelihood value for a fixed y vector
  while(d > del && k <= iter){
    if(trace) cat("\niteration", k, "nrounds", nrounds, "\n") 
    if(k==1) weight_update <- weights else
    weight_update <- mpath::compute_wt(ylos, weights, cfunval, s, delta)
    weight_update_log[,k] <- weight_update
    #if(trace) cat("weight_update", weight_update, "\n")
    RET <- xgboost::xgboost(data=x, label=y, missing = NA, weight=weight_update, params = params, nrounds=nrounds, ...)
    ypre <- predict(RET, x) #depends on objective, this is probability or response or linear predictor
    #update loss values
    if(dfun=="reg:squarederror"){
      ylos <- (ynew - ypre)^2/2
    }else if(dfun=="binary:logitraw"){
    #u <- 1/(1+exp(-ypre)) # for y in [0, 1]
    #ylos <- -y*log(u/(1-u)) - log(1-u)
    ylos <- log(1 + exp( - ynew * ypre)) # for y in [-1, 1], the results ylos should be the same
    }else if(dfun=="binary:hinge"){
      ylos <- pmax(0, 1- ynew * ypre)
    }else if(dfun=="multi:softprob"){
      num_class <- RET$params$num_class
      # reshape it to a num_class-columns matrix
      ypre <- matrix(ypre, ncol=num_class, byrow=TRUE)
      ylos <- rep(NA, n)
      for(i in 1:n)
        ylos[i] = - log(ypre[i, y[i]+1]) # label y is coded as in [0, num_class-1]
    }else if(dfun %in% c("count:poisson")){
      ylos <- loss3(ynew, mu=ypre, theta=1, weights, cfunval, family=3, s, delta)$z
    }else if(dfun %in% c("reg:gamma")){
      ylos <- y/ypre+log(ypre) #negative log-likelihood value with "parameter"=1 in xgboost
      ylos <- ylos - min_nloglik #to shift the values to non-negative
    }else if(dfun %in% c("reg:tweedie")){
        #extract tweedie_variance_power
        rho <- substring(names(RET$evaluation_log[2]), 23)
        rho <- as.numeric(rho)
        a <- y * exp((1-rho)*log(ypre))/(1-rho)
        b <-     exp((2-rho)*log(ypre))/(2-rho)
        ylos <- - a + b
    }
    loss_log[k] <- sum(mpath::compute_g(ylos, cfunval, s, delta))
    if(k > 1){
      d <- abs((loss_log[k-1]-loss_log[k]))/loss_log[k-1]
      if(loss_log[k] > loss_log[k-1])
        nrounds <- nrounds + 100	    
    }
    if(trace) cat("loss=", loss_log[k], "d=", d, "\n") 
    k <- k + 1
  }
  RET$x <- x
  RET$y <- y
  RET$call <- call
  RET$weight_update_log <- weight_update_log
  RET$weight_update <- weight_update
  RET$loss_log <- loss_log
  RET
}
