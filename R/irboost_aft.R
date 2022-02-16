#' fit a robust accelerated failure time model with iteratively reweighted boosting algorithm

#'
#' Fit an accelerated failure time model with the iteratively reweighted convex optimization   (IRCO) that minimizes the robust loss functions in the CC-family (concave-convex).     The convex optimization is conducted by functional descent boosting algorithm   in the R package \pkg{xgboost}. The iteratively reweighted boosting (IRBoost) algorithm reduces the weight of the        observation that leads to a large loss; it also provides weights to help        identify outliers. For time-to-event data, an accelerated failure time model (AFT
#' model) provides an alternative to the commonly used proportional hazards models. Note, \code{irboost} with \code{dfun=survival:aft} is the wrapper of \code{irboost_aft}, which was developed to facilitate a different data input format used in \code{xgb.train} not in \code{xgboost} at the time.
#' @param params the list of parameters used in \code{xgb.train} of \pkg{xgboost}. Must include \code{aft_loss_distribution}, \code{aft_loss_distribution_scale}, but there is no need to include \code{objective}. The complete list of parameters is
#'   available in the \href{http://xgboost.readthedocs.io/en/latest/parameter.html}{online documentation}.
#' @param data training dataset. \code{irboost_aft} accepts only an \code{xgb.DMatrix} as the input.
#' @param cfun concave component of CC-family, can be \code{"hacve", "acave", "bcave", "ccave"}, 
#' \code{"dcave", "ecave", "gcave", "hcave"}. See Table 2 in https://arxiv.org/pdf/2010.02848.pdf
#' @param s tuning parameter of \code{cfun}. \code{s > 0} and can be equal to 0 for \code{cfun="tcave"}. If \code{s} is too close to 0 for                     \code{cfun="acave", "bcave", "ccave"}, the calculated weights can become 0 for all observations, thus crash the program
#' @param delta a small positive number provided by user only if \code{cfun="gcave"} and \code{0 < s <1}
#' @param iter number of iteration in the IRCO algorithm
#' @param nrounds boosting iterations in \code{xgb.train} within each IRCO iteration
#' @param del convergency criteria in the IRCO algorithm, no relation to \code{delta}
#' @param trace if \code{TRUE}, fitting progress is reported
#' @param ... other arguments passing to \code{xgb.train} 
#' @importFrom stats predict
#' @importFrom xgboost xgb.train getinfo setinfo
#' @importFrom mpath compute_wt compute_g cfun2num
#' @return An object of class \code{xgb.Booster} with additional elements:
#' \itemize{
#'   \item \code{weight_update_log} a matrix of \code{nobs} row by {iter} column of observation weights in each iteration of the IRCO algorithm
#'   \item \code{weight_update} a vector of observation weights in the last IRCO iteration that produces the final model fit
#'   \item \code{loss_log} sum of loss value of the composite function \code{cfun(survival_aft_distribution)} in each IRCO iteration
#' }
#'
#' @examples
#' \donttest{
#' library("xgboost")
#' X <- matrix(1:5, ncol=1)
#'
#' # Associate ranged labels with the data matrix.
#' # This example shows each kind of censored labels.
#' #                   uncensored  right  left  interval
#' y_lower = c(10,  15, -Inf, 30, 100)
#' y_upper = c(Inf, Inf,   20, 50, Inf)
#' dtrain <- xgb.DMatrix(data=X, label_lower_bound=y_lower, label_upper_bound=y_upper)
#'                       params = list(objective="survival:aft", aft_loss_distribution="normal",
#'                       aft_loss_distribution_scale=1, max_depth=3, min_child_weight= 0)
#' watchlist <- list(train = dtrain)
#' bst <- xgb.train(params, dtrain, nrounds=15, watchlist=watchlist)
#' predict(bst, dtrain)
#' bst_cc <- irboost_aft(params, dtrain, nrounds=15, watchlist=watchlist, cfun="hcave", 
#'                       s=1.5, trace=TRUE, verbose=0)
#' bst_cc$weight_update
#' predict(bst_cc, dtrain)
#' }
#'
#' @seealso
#' \code{\link{irboost}}
#'
#' @author Zhu Wang\cr Maintainer: Zhu Wang \email{zhuwang@gmail.com}
#' @references Wang, Zhu (2021), \emph{Unified Robust Boosting}, arXiv eprint, \url{https://arxiv.org/abs/2101.07718}
#' @keywords regression survival
#' @export irboost_aft

irboost_aft <- function(params, data, cfun="ccave", s=1, delta=0.1, iter=10, nrounds=100, del=1e-10, trace=FALSE, ...){
   call <- match.call()
   params$objective <- "survival:aft"
   if(params$objective!="survival:aft") warnings("params$objective is supposed to be survival:aft")
   cfunval <- eval(parse(text="mpath::cfun2num(cfun)"))
   d <- 10
   k <- 1
   if(trace) {
      cat("\nrobust boosting ...\n")
   }   
   n <- getinfo(data, "nrow")
   loss_log <- rep(NA, iter)
   weight_update_log <- matrix(NA, nrow=n, ncol=iter)
   y_lower <- getinfo(data, "label_lower_bound")
   y_upper <- getinfo(data, "label_upper_bound")
   weights <- getinfo(data, "weight")
   if(is.null(weights)) weights <- rep(1, n)
   ylos <- weights #initial values
   while(d > del && k <= iter){
      if(trace) cat("\niteration", k, "nrounds", nrounds, "\n")
      if(k==1) weight_update <- weights else
         weight_update <- mpath::compute_wt(ylos, weights, cfunval, s, delta)
      weight_update_log[,k] <- weight_update
      if(trace) cat("weight_update", weight_update, "\n")
      setinfo(data, 'weight', weight_update) #update data weight 
      RET <- xgb.train(params, data, nrounds=nrounds, ... )
      ypre <- log(predict(RET, data)) #aftloss function is based on log tranformed prediction
      for(i in 1:n)
      ylos[i] <- aftloss(y_lower[i], y_upper[i], ypre[i], sigma=params$aft_loss_distribution_scale,       distribution=params$aft_loss_distribution)
      loss_log[k] <- sum(mpath::compute_g(ylos, cfunval, s, delta))
      if(k > 1){
         d <- abs((loss_log[k-1]-loss_log[k]))/loss_log[k-1]
         if(loss_log[k] > loss_log[k-1])
            nrounds <- nrounds + 100
      }
      if(trace) cat("loss=", loss_log[k], "d=", d, "\n")
      k <- k + 1
   }
   RET$call <- call
   RET$weight_update_log <- weight_update_log
   RET$weight_update <- weight_update
   RET$loss_log <- loss_log
   RET
}

