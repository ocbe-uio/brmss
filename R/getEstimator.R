#' @title Extract the posterior estimate of parameters
#' @description
#' Extract the posterior estimate of the parameters of a \code{brmss} class object.
#' @name getEstimator
#'
#' @param object an object of class \code{brmss}
#' @param estimator the name of one estimator. Default is the latent indicator
#' estimator "\code{gamma}". Other options are among
#' "\code{c('beta', 'elpd', 'logP')}"
#' @param Pmax threshold that truncate the estimator "\code{gamma}" or
#' "\code{eta}". Default is \code{0}. If \code{Pmax=0.5} and
#' \code{type="conditional"}, it gives median probability model betas
#' @param type the type of output beta. Default is \code{marginal}, giving
#' marginal beta estimation. If \code{type="conditional"}, it gives beta
#' estimation conditional on gamma=1
#'
#' @return Return the estimator from an object of class \code{brmss}. It is
#' a matrix or vector
#'
#' @examples
#'
#' # simulate data
#' set.seed(123)
#' n <- 200 # subjects
#' p <- 10 # variable selection predictors
#'
#' dat <- simData(n, p, model = "weibull")
#'
#' # run a Bayesian brmss model
#' fit <- brmss(dat$y, dat$x, family = "weibull", nIter = 10, burnin = 1)
#'
#' gamma.hat <- getEstimator(fit, estimator = "gamma")
#'
#' @export
getEstimator <- function(object, estimator = "gamma", Pmax = 0,
                         type = "marginal") {
  if (!estimator %in% c("gamma", "beta", "elpd", "logP")) {
    stop("Please specify correct 'estimator'!")
  }

  if (Pmax < 0 || Pmax > 1) {
    stop("Please specify correct argument 'Pmax' in [0,1]!")
  }

  if (!type %in% c("conditional", "marginal")) {
    message("NOTE: The argument type is invalid!")
  }
  if (Pmax < 0 || Pmax > 1) {
    message("NOTE: The argument Pmax is invalid!")
  }
  
  burnin <- object$input$burnin / object$input$thin
  L <- NCOL(object$input$y)

  if (estimator %in% c("gamma", "beta")) {
    estimate <- matrix(colMeans(object$output$gammas[-c(1:burnin), ]), ncol = L)
    # estimate <- object$output$post$gammas
    # estimate <- rbind(1, estimate)
    gammas <- estimate
    if (Pmax > 0) {
      estimate[estimate <= Pmax] <- 0
      estimate[estimate > Pmax] <- 1
    }
  }

  if (estimator == "beta") {
    estimate <- matrix(colMeans(object$output$betas[-c(1:burnin), ]), ncol = L)
    # estimate <- object$output$post$betas

    if (type %in% c("marginal", "conditional")) {
      if (type == "conditional") {
        estimate <- (gammas >= Pmax) * estimate / gammas
        estimate[is.na(estimate)] <- 0
      }
    } else {
      stop("Please specify correct type!")
    }
  }

  if (estimator == "elpd") {
    estimate <- loo::loo(object$output$loglikelihood[
      1 + (object$input$burnin:object$input$nIter) / object$input$thin,
    ])
    estimate
  }

  if (estimator == "logP") {
    estimate <- rowSums(object$output$loglikelihood)
  }

  return(estimate)
}
