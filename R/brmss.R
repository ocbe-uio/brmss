#' @title Fit Bayesian Regression Models
#'
#' @description
#' This is the main function to fit various Bayesian regression models
#'
#' @name brmss
#'
#' @importFrom Rcpp evalCpp
#'
#' @param y response variable. A vector, matrix or dataframe. For survival 
#' outcomes, the first column of \code{y} is times and the second column is 
#' events
#' @param x input matrix or dataframe
#' @param family a character string representing one of the built-in families
#' @param nIter the number of iterations of the chain
#' @param burnin number of iterations to discard at the start of the chain
#' @param thin thinning MCMC intermediate results to be stored
#' @param tick an integer used for printing the iteration index and some updated
#' parameters every tick-th iteration. Default is 1
#' @param hyperpar a list of relevant hyperparameters
#' @param threads number of threads used for parallelization. Default is 1
#' @param gammaSampler one of \code{c("mc3", "bandit")}
#' @param gammaProposal one of \code{c("simple", "posterior")}
#' @param gammaGibbs one of \code{c("none", "independent", "gprior")}. If 
#' \code{gammaGibbs = "independent"}, it implements the approach in 
#' Kuo and Mallick (1998) with gibbs sampling for gammas and with independent 
#' spike-and-slab priors for betas. If \code{gammaGibbs = "gprior"}, it 
#' implements the approach in George and McCulloch (1997) with gibbs sampling 
#' for gammas and with g-prior for betas independent spike-and-slab priors for 
#' betas.
#' @param varPrior string indicating the prior for the variance of 
#' response/error term. Default is \code{IG}. For continous multivariate 
#' responses, it can also be \code{IW} for the inverse-Wishart prior (dense 
#' variance-covariance matrix) or \code{HIW} for the hyper-inverse Wishart 
#' (sparse precision matrix)
#' @param initial a list of initial values for parameters "kappa" and "betas"
#' @param arms.list a list of parameters for the ARMS algorithm
#'
#'
#' @return An object of a list including the following components:
#' \itemize{
#' \item input - a list of all input parameters by the user
#' \item output - a list of the all mcmc output estimates:
#' \itemize{
#' \item "\code{kappa}" - a vector with MCMC intermediate estimates of the Weibull's shape parameter
#' \item "\code{betas}" - a matrix with MCMC intermediate estimates of effects on cluster-specific survival
#' \item "\code{gammas}" - a matrix with MCMC intermediate estimates of inclusion indicators of variables for cluster-specific survival
#' \item "\code{gamma_acc_rate}" - acceptance rate of the M-H sampling for gammas
#' \item "\code{loglikelihood}" - a matrix with MCMC intermediate estimates of individuals' likelihoods
#' \item "\code{tauSq}" - a vector with MCMC intermediate estimates of tauSq
#' \item "\code{sigmaSq}" - a vector with MCMC intermediate estimates of sigmaSq
#' \item "\code{post}" - a list with posterior means of "kappa", "betas", "gammas" and sigmaSq"
#' }
#' \item call - the matched call
#' }
#'
#' @references Eddelbuettel D, Sanderson C (2014). \emph{RcppArmadillo: Accelerating R with high-performance C++ linear algebra}. Computational Statistics and Data Analysis, 71, 1054--1063
#' @references Zhao Z (2025+). arXiv.
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
#' fit <- brmss(dat$y, dat$x, family = "weibull", nIter = 100, burnin = 10)
#'
#' @export
brmss <- function(y, x,
                  family = "weibull",
                  nIter = 500,
                  burnin = 200,
                  thin = 1,
                  tick = 100,
                  hyperpar = NULL,
                  threads = 1,
                  gammaSampler = "bandit",
                  gammaProposal = "simple",
                  gammaGibbs = "none",
                  varPrior = "IG",
                  initial = NULL,
                  arms.list = NULL) {
  # Validation
  stopifnot(burnin < nIter)
  stopifnot(burnin >= 0)
  
  if (!family %in% c("gaussian", "logit", "probit", "weibull", 
                     "dirichlet", "mgaussian", "mvprobit")) {
    stop('Argument "family" is not valid!')
  }
  
  if (!is.matrix(y)) {
    y <- as.matrix(y)
  }

  if (family == "weibull") {
    L <- 1
  } else {
    L <- NCOL(y)
  }

  if (!is.matrix(x)) {
    x <- as.matrix(x)
  }

  if (is.null(arms.list)) {
    arms.list <- list(
      n = 1, # This should always be n=1 with the current code
      nsamp = 1,
      ninit = 10,
      convex = 1,
      npoint = 100
    )
    # sampling" or the "derivative-free adaptive rejection sampling with
    # metropolis step" (default) is used
  }

  if (arms.list$n != 1) {
    stop("Need to modify 'arms_gibbs.cpp' if arms.list$n > 1!")
  }

  # check the formula
  cl <- match.call()

  gammaSampler <- tolower(gammaSampler)
  if (!gammaSampler %in% c("mc3", "bandit")) {
    stop('Argument "gammaSampler" must be one of c("mc3", "bandit")!')
  }
  
  gammaGibbs <- tolower(gammaGibbs)
  if (!gammaGibbs %in% c("none", "independent", "gprior")) {
    stop('Argument "gammaGibbs" must be one of c("none", "independent", "gprior")!')
  }
  
  varPrior <- toupper(varPrior)
  if (!varPrior %in% c("IG", "IW", "HIW")) {
    stop('Argument "varPrior" must be one of c("IG", "IW", "HIW")!')
  }

  # set hyperparamters of all piors
  # if (is.null(hyperpar)) {
  if (is.null(hyperpar)) {
    hyperpar <- list()
  }

  # if (!"pi" %in% names(hyperpar)) {
  #   hyperpar$pi <- 0 # TODO: enable this if not use a hyperprior for 'pi' 
  # }
  
  # beta-Bernoulli's hyperparameters
  if (!"piA" %in% names(hyperpar)) {
    hyperpar$piA <- 1
    hyperpar$piB <- NCOL(x)
  }
  
  if (!"sigmaA" %in% names(hyperpar)) {
    hyperpar$sigmaA <- 5
    hyperpar$sigmaB <- 20
  }
  
  # hyperpar$tauA <- 20; hyperpar$tauB <- 50
  if (!"tauA" %in% names(hyperpar)) {
    hyperpar$tauA <- 5
    hyperpar$tauB <- 20
  }
  if (!"tau0A" %in% names(hyperpar)) {
    hyperpar$tau0A <- hyperpar$tauA
    hyperpar$tau0B <- hyperpar$tauB
  }

  if (!"kappaA" %in% names(hyperpar)) {
    hyperpar$kappaA <- 1 # 3
    hyperpar$kappaB <- 1 # This is for Gamma prior
  }
  
  if (!"pj" %in% names(hyperpar)) {
    hyperpar$pj <- 0.5 # This is fixed Bernoulli probability according to Kuo & Mallick (1998, Sankhyā)
  }
  
  if (!"nu" %in% names(hyperpar)) {
    hyperpar$nu <- L + 2 
  }
  
  if (!"psiA" %in% names(hyperpar)) {
    hyperpar$psiA <- 0.1
    hyperpar$psiB <- 10
  }
  
  # hyperpar$tauSq <- rep(1, L)
  # hyperpar$tau0Sq <- 1

  # initialization of parameters
  if (is.null(initial)) {
    initList <- list()

    initList$kappa <- 0.9
    initList$betas <- matrix(0, nrow = NCOL(x) + 1, ncol = L) 
    initList$betas[1] <- 0.1 # initial intercept 0.1
    initList$gammas <- matrix(0, nrow = NCOL(x) + 1, ncol = L) 
  }

  if (!"bound.pos" %in% names(hyperpar)) {
    hyperpar$bound.neg <- -10
    hyperpar$bound.pos <- 10
  }
  hyperpar$bound.kappa <- 1e-2
  rangeList <- list(
    betaMin = hyperpar$bound.neg, betaMax = hyperpar$bound.pos,
    kappaMin = hyperpar$bound.kappa, kappaMax = hyperpar$bound.pos
  )


  #################
  ## Output objects
  #################

  ret <- list(input = list(), output = list(), call = cl)
  class(ret) <- "brmss"

  ret$input$y <- y
  ret$input$x <- x
  ret$input$family <- family
  ret$input$nIter <- nIter
  ret$input$burnin <- burnin
  ret$input$thin <- thin
  ret$input$hyperpar <- hyperpar
  ret$input$arms.list <- arms.list


  #################
  ## Main steps for Bayesian inference
  #################

  ## MCMC iterations
  ret$output <- run_mcmc(
    y,
    x,
    family,
    nIter,
    burnin,
    thin,
    tick,
    gammaSampler,
    gammaProposal,
    gammaGibbs,
    varPrior,
    threads,
    arms.list$n, # n: number of samples to draw, now only 1
    arms.list$nsamp, # nsamp: number of MCMC for generating each ARMS sample, only keeping the last one
    arms.list$ninit, # ninit: number of initials as meshgrid values for envelop search
    arms.list$convex, # convex: adjustment for convexity
    arms.list$npoint, # npoint: maximum number of envelope points
    initList,
    rangeList,
    hyperpar
  )

  return(ret)
}
