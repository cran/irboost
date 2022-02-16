#' @importFrom stats pnorm

pdf <- function(z, distribution){
  kPI = 3.14159265358979323846;
  if(distribution=="normal"){
    return(exp(-z * z / 2.0) / sqrt(2.0 * kPI))
  }else if(distribution=="logistic"){
    w = exp(z);
    sqrt_denominator = 1 + w;
    if (is.infinite(w) || is.infinite(w * w)) {
      return(0)
    } else {
      return(w / (sqrt_denominator * sqrt_denominator))
    }
  }else if(distribution=="extreme"){
    w = exp(z);
    if(is.infinite(w)) return(0) else return(w * exp(-w))     
  }
}
cdf <- function(z, distribution){
  if(distribution=="normal"){
    #return(0.5 * (1 + erf(z / sqrt(2.0))))
      return(stats::pnorm(z))
  }else if(distribution=="logistic"){
    w = exp(z);
    if(is.infinite(w)) return(1) else return(w / (1 + w))
  }else if(distribution=="extreme"){
    w = exp(z);
    return(1 - exp(-w))
  }
}

