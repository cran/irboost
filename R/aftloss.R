aftloss <- function(y_lower, y_upper, y_pred, sigma, distribution){
kEps = 1e-12;  # A denominator in a fraction should not be too small
    log_y_upper = log(y_upper);
    if (y_lower == y_upper) {  # uncensored
      z = (log(y_lower) - y_pred) / sigma;
      pdf = pdf(z, distribution)
      # Regularize the denominator with eps, to avoid INF or NAN
      cost = -log(max(pdf / (sigma * y_lower), kEps));
    } else {  # censored; now check what type of censorship we have
      if (is.infinite(y_upper)) {  # right-censored
        cdf_u = 1;
      } else {  # left-censored or interval-censored
        z_u = (log_y_upper - y_pred) / sigma;
        cdf_u = cdf(z_u, distribution);
      }
      if (y_lower <= 0.0) {  # left-censored
        cdf_l = 0;
      } else {  # right-censored or interval-censored
        z_l = (log(y_lower) - y_pred) / sigma;
        cdf_l = cdf(z_l, distribution);
      }
      # Regularize the denominator with eps, to avoid INF or NAN
      cost = -log(max(cdf_u - cdf_l, kEps));
    }
    return(cost)
  }
