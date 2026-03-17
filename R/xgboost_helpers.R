# Internal helpers for xgboost interop.

.reshape_softprob <- function(pred, n, num_class) {
  # xgboost has changed the ordering of multi:softprob predictions across
  # versions. Try both reshapes and pick the one whose rows sum closest to 1.
  if (length(pred) != n * num_class) {
    stop("Unexpected multi:softprob prediction length: got ", length(pred),
         ", expected ", n * num_class)
  }

  m_byrow <- matrix(pred, ncol = num_class, byrow = TRUE)
  m_bycol <- matrix(pred, ncol = num_class, byrow = FALSE)

  score <- function(m) max(abs(rowSums(m) - 1), na.rm = TRUE)
  if (is.finite(score(m_bycol)) && score(m_bycol) < score(m_byrow)) m_bycol else m_byrow
}
