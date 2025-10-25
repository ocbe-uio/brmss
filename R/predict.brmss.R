#' @title Prediction of survival probability
#'
#' @description
#' Compute predicted survival probability for a brmss
#'
#' @name predict.brmss
#'
#' @param object the results of a \code{brmss} fit
#' @param dat the dataset used in \code{brmss()}
#' @param newdata optional new data at which to do predictions. If missing, the
#' prediction will be based on the training data
#' @param type the type of predicted value. Currently it is only valid with
#' \code{'survival'}
#' @param times evaluation time points for survival prediction. Default
#' \code{NULL} for predicting all time points in the \code{newdata} set
#' @param ... for future methods
#'
#' @return A matrix object
#'
#'
#' @examples
#'
#' # simulate data
#' set.seed(123)
#' n <- 200 # subjects
#' p <- 10 # variable selection predictors
#'
#' dat <- simData(n, p)
#'
#' # run a Bayesian brmss model: brmss-Ber2
#' fit <- brmss(dat$y, dat$x, family = "weibull", nIter = 100, burnin = 10)
#'
#' # pred.survival <- predict(fit, dat, newdata = dat, times = c(1, 3, 5))
#'
#' @export
predict.brmss <- function(object, dat,
                          newdata = NULL,
                          type = "survival",
                          times = NULL, ...) {
  n <- NROW(dat$x)
  if (object$input$family == "weibull") {
    L <- 1
  } else {
    L <- NCOL(dat$y)
  }

  survObj <- data.frame(dat$y, dat$x)

  if (is.null(dat.new)) {
    dat.new.flag <- FALSE
    dat.new <- dat
    survObj.new <- survObj
  } else {
    dat.new.flag <- TRUE
    survObj.new <- data.frame(dat.new$y, dat.new$x)
  }

  burnin <- object$input$burnin / object$input$thin

  # survival predictions based on posterior mean
  betas.hat <- matrix(colMeans(object$output$betas[-c(1:burnin), ]), ncol = L)
  gammas.hat <- matrix(colMeans(object$output$gammas[-c(1:burnin), ]), ncol = L)
  gammas.hat <- rbind(1, gammas.hat)
  betas.hat <- (gammas.hat >= 0.5) * betas.hat / gammas.hat
  betas.hat[is.na(betas.hat)] <- 0



  if (object$input$family == "weibull") {
    # predict survival probabilities based on brmss
    time_eval <- sort(times)
    if (is.null(time_eval)) {
      time_eval <- sort(survObj.new$time)
    }
    kappa.hat <- mean(object$output$kappa[-c(1:burnin)])

    # predict survival probabilities based on brmss
    Surv.prob <- matrix(nrow = n, ncol = length(time_eval))

    mu <- exp(cbind(1, dat.new$x) %*% betas.hat)
    lambdas <- mu / gamma(1 + 1 / kappa.hat)
    for (j in seq_along(time_eval)) {
      Surv.prob[, j] <- exp(-(time_eval[j] / lambdas)^kappa.hat)
    }
  }

  return(Surv.prob)
}
