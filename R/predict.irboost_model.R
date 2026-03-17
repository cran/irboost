#' @export
predict.irboost_model <- function(object, newdata, ...) {
  # Forward to the underlying xgboost model, but normalize the output
  # ordering for multi:softprob across xgboost versions.
  pred <- predict(object$model, newdata = newdata, ...)

  # Detect multi:softprob output by length: for multiclass it is n * num_class.
  # This avoids relying on xgboost helper APIs that have changed across versions.
  n <- if (inherits(newdata, "xgb.DMatrix")) xgboost::getinfo(newdata, "nrow") else nrow(newdata)
  if (!is.null(n) && length(pred) > n && (length(pred) %% n) == 0L) {
    num_class <- as.integer(length(pred) / n)
    m <- .reshape_softprob(pred, n = n, num_class = num_class)
    # Return in the historical "by observation" (row-major) order:
    # obs1: p(class0..k-1), obs2: ...
    pred <- as.vector(t(m))
  }

  pred
}
