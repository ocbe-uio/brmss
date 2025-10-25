#' @title Metropolis sampler for a target density
#'
#' @description
#' Random number generator via Metropolis-Hastings algorithm.
#'
#' @name metropolis_sampler
#'
#' @importFrom stats rweibull runif
#'
#' @param initial_value initial values
#' @param n number of draws
#' @param proposal_shape Weibull's shape parameter in the proposal
#' @param proposal_scale Weibull's scale parameter in the proposal
#' @param mu mean survival time
#' @param kappas Weibull's true shape parameter
#' @param burnin length of burn-in period
#' @param lag discarding lag-1 values in the Metropolis step
#'
#' @return A dataframe consisting of the sampled values and acceptance rate
#'
#'
#' @examples
#'
#' times <- metropolis_sampler(10, 5)
#'
#' @export
metropolis_sampler <- function(initial_value,
                               n = n,
                               proposal_shape = 1,
                               proposal_scale = 1,
                               mu = 1,
                               kappas = 0.9,
                               burnin = 0,
                               lag = 1) {
  results <- list()
  current_state <- initial_value
  for (i in 1:burnin) {
    out <- metropolis_step(
      current_state, proposal_shape, proposal_scale,
      mu, kappas
    )
    if (!is.na(out$value)) {
      current_state <- out$value
    }
  }
  for (i in 1:n) {
    for (j in 1:lag) {
      out <- metropolis_step(
        current_state, proposal_shape, proposal_scale,
        mu, kappas
      )
      if (!is.na(out$value)) {
        current_state <- out$value
      }
    }
    results[[i]] <- out
  }
  results <- do.call(rbind, results)
  results
}

## internal function for the Metropolis-Hastings algorithm
metropolis_step <- function(x, proposal_shape, proposal_scale,
                            mu, kappas) {
  proposed_x <- rweibull(1, shape = proposal_shape, scale = proposal_scale)
  accept_prob <- min(1, target(proposed_x, mu, kappas) /
    target(x, mu, kappas))
  u <- runif(1)
  if (is.na(u <= accept_prob)) {
    value <- NA
    accepted <- FALSE
  } else {
    if (u <= accept_prob) {
      value <- proposed_x
      accepted <- TRUE
    } else {
      value <- x
      accepted <- FALSE
    }
  }
  out <- data.frame(value = value, accepted = accepted)
  out
}
