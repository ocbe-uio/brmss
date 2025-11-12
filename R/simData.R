#' @title Simulate data
#'
#' @description
#' Simulate survival data based on a Cox model
#'
#' @name simData
#'
#' @importFrom stats rbinom rnorm runif rexp rgamma pnorm
#'
#' @param n number of subjects
#' @param p number of covariates in each cluster
#' @param L number of response variables
#' @param kappas value of the Weibull's shape parameter
#' @param model one of \code{c("gaussian", "logit", "probit", "mgaussian", "dirichlet", "cox")}
#'
#' @return An object of a list
#' \itemize{
#' \item "\code{x}" - an matrix of covariates
#' \item "\code{y}" - a dataframe including events and times
#' \item "\code{betas}" - covariate effects
#' \item "\code{kappas}" - value of the Weibull's shape parameter
#' }
#'
#'
#' @examples
#'
#' # simulate data
#' set.seed(123)
#' n <- 200 # subjects
#' p <- 10 # variable selection predictors
#' dat <- simData(n, p, model = "weibull")
#' str(dat)
#'
#' @export
simData <- function(n = 200, p = 10, L = 1,
                    kappas = 2,
                    model = "weibull") {
  ## predefined functions
  Expo <- function(times, surv) {
    z1 <- -log(surv[1])
    t1 <- times[1]
    lambda <- z1 / (t1)
    list(rate = lambda)
  }

  ## effects
  if (p < 5) {
    stop("p must be at least 5!")
  }
  betas <- matrix(c(-1, -.5, .8, .8, -1, rep(0, p * L - 5)), nrow = p, ncol = L)

  # generate other effects if multi-responses
  if (L > 1) {
    betas[1:min(2, p), 2:L] <- c(-1, 1) # runif(min(2, p) * (L - 3), -2, 2)
  }

  # add intercept
  betas <- rbind(0.5, betas)

  ## covariates
  x <- scale(mvnfast::rmvn(n, rep(0, p), diag(p)))
  attr(x, "scaled:center") <- attr(x, "scaled:scale") <- NULL
  colnames(x) <- paste0("X", 1:p)
  
  ## simulate (multivariate) Gaussian responses 
  if (model %in% c("gaussian", "mgaussian")) {
    y <- cbind(1, x) %*% betas + rnorm(n * L)
    dat <- list(y = y, x = x, betas = betas)
  }
  
  # ## simulate (multivariate) probit binary responses 
  if (model == "probit") {
    y <- matrix(NA, nrow = n, ncol = L)
    betas[1, ] <- -0.5
    repeat{
        y <- matrix(rbinom(n * L, 1, pnorm( cbind(1, x) %*% betas )), ncol = L)
      
      if( min(colMeans(y)) > 0.01 && max(colMeans(y)) < 1 - 0.01 )
        break
      
      betas[1, ] <- betas[1, ] * (1 - 0.1)
    }
    dat <- list(y = y, x = x, betas = betas)
  }
  
  # ## simulate logistic binary responses 
  if (model == "logit") {
    if (L != 1) {
      stop("Simiulating logistic model responses requires L=1!")
    }
    y <- matrix(NA, nrow = n, ncol = 1)
    betas[1] <- -0.5
    repeat{
      y[, 1] <- rbinom(n, 1, 1 / (1 + exp(-cbind(1, x) %*% betas)))
      if( mean(y) > 0.10 && mean(y) < 1 - 0.10)
        break
      
      betas[1] <- betas[1] * (1 - 0.1)
    }
    dat <- list(y = y, x = x, betas = betas)
  }
  
  ## simulate proportions from Dirichlet distribution (n, alpha=1:L)
  if (model == "dirichlet") {
    if (L == 1) {
      stop("To simulate Dirichlet data, argument 'L' must be at least 2!")
    }
    
    alphas <- matrix(nrow = n, ncol = L)
    for (l in 1:L) {
      alphas[, l] <- exp(cbind(1, x) %*%
        matrix(betas[, l], ncol = 1))
    }
    # alpha0 <- alpha1 + alpha2 + alpha3
    # proportion <- cbind(alpha1 / alpha0, alpha2 / alpha0, alpha3 / alpha0)

    ## Generate n-individual Dirichlet proportions
    proportion <- matrix(nrow = n, ncol = L)
    for (i in 1:n) {
      proportion[i, ] <- sapply(1:L, function(l) rgamma(1, alphas[i, l]))
    }

    # proportion <- matrix(rgamma(L * n, t(1:L)), ncol = L, byrow=TRUE)
    y <- proportion / rowSums(proportion)
    colnames(y) <- paste0("Y", 1:L)
    
    dat <- list(y = y, x = x, betas = betas)
  }

  if (model == "weibull") {
    # censoring function
    # - follow-up time 1 to 3 years
    # - administrative censoring: uniform data entry (cens1)
    # - loss to follow-up: exponential, 20% loss (cens2)
    ACT <- 1
    FUT <- 3 # 5
    cens.start <- FUT
    cens.end <- ACT + FUT
    cens1 <- runif(n, cens.start, cens.end)
    loss <- Expo(times = 5, surv = 0.8) # 0.6)#
    cens2 <- rexp(n, rate = loss$rate)
    cens <- pmin(cens1, cens2)

    ## simulate event times by M-H algorithm
    T.star <- cens
    accepted <- numeric(n)
    mu0 <- exp(cbind(1, x) %*% betas)


    for (i in 1:n) {
      ## M-H sampler for event time
      # If the target is set as Gompertz distr., it's a bit model misspecification
      out <- metropolis_sampler(
        initial_value = 10,
        n = 5,
        proposal_shape = 0.9,
        proposal_scale = mean(cens),
        mu = mu0[i],
        kappas = kappas,
        burnin = 100,
        lag = 10
      )
      T.star[i] <- mean(out$value)
      accepted[i] <- mean(out$accepted)
    }
    # survival object
    event <- ifelse(T.star <= cens, 1, 0) # censoring indicator
    times <- pmin(T.star, cens) # observed times
    y <- data.frame(time = times, event = event)
    
    dat <- list(y = y, x = x, betas = betas, kappa = kappas)
  }

  if (model == "cox") {
    # simulate event times from Weibull distribution (WEI2)
    T.star <- (-log(runif(n)) * (1 / kappas) *
      exp(-x %*% betas))^(1 / kappas)

    # survival object
    event <- ifelse(T.star <= cens, 1, 0) # censoring indicator
    times <- pmin(T.star, cens) # observed times
    y <- data.frame(event = event, time = times)
    
    dat <- list(y = y, x = x, betas = betas, kappa = kappas)
  }
  
  if (L > 1) {
    colnames(dat$y) <- paste0("Y", 1:L)
  }

  return(dat)
}
