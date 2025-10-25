#' @title Target density
#'
#' @description
#' Predefined target density corresponding to the population survival function
#' of brmss
#'
#' @name target
#'
#' @param x value generated from the proposal distribution
#' @param mu mean survival time
#' @param kappas Weibull's true shape parameter
#'
#' @return value of the targeted (improper) probability density function
#'
#'
#' @examples
#'
#' time1 <- target(1.2, 0.1, 2)
#'
#' @export
target <- function(x, mu, kappas) {
  ## Weibull 3
  lambdas <- mu / gamma(1 + 1 / kappas)
  # survival.function <- exp(-(x / lambdas)^kappas)
  # improper pdf
  pdf <- kappas / lambdas * (x / lambdas)^(kappas - 1) *
    exp(-(x / lambdas)^kappas)

  return(pdf)
}
