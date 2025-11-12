#' @title MCMC trace-plots
#'
#' @description
#' Trace-plots of regression coefficients over MCMC iterations
#'
#' @name plotMCMC
#'
#' @importFrom ggplot2 ggplot aes geom_step theme element_blank
#' @importFrom graphics segments
#'
#' @param dat input data as a list containing outcomes and covariates
#' @param datMCMC returned object from the main function \code{brmss()}
#' @param estimator print estimators, one of
#' \code{c("beta", "gamma", "tau", "sigma", "kappa")}
#'
#' @return A \code{ggplot2::ggplot} object. See \code{?ggplot2::ggplot} for more
#' details of the object.
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
#' plotMCMC(dat, datMCMC = fit, estimator = "tau")
#'
#' @export
plotMCMC <- function(dat, datMCMC, estimator = "tau") {
  oldpar <- par(no.readonly = TRUE)
  on.exit(par(oldpar))

  p <- NCOL(dat$x)
  if (datMCMC$input$family == "weibull") {
    L <- 1
  } else {
    L <- NCOL(dat$y)
  }

  p.idx <- 1:((p + 1) * L)
  if (p > 10) {
    p <- 10
    p.idx <- rep(1:11, L) + (p + 1) * rep(0:(L - 1), each = 11)
  }
  if ("beta" %in% estimator) {
    betas.mcmc <- datMCMC$output$betas[, p.idx]
    dat$betas <- rbind(dat$beta0, dat$betas)
    ylabel <- paste0(
      "expression(beta['", rep(0:p, L), ",",
      rep(1:L, each = p + 1), "'])"
    )
    layout(matrix(seq_len(NCOL(betas.mcmc)), ncol = L))
    par(mar = c(2, 4.1, 2, 2))
    for (j in seq_len(NCOL(betas.mcmc))) {
      plot(betas.mcmc[, j],
        type = "l", lty = 1, ylab = eval(parse(text = ylabel[j])),
        ylim = summary(c(betas.mcmc, dat$betas))[c(1, 6)]
      )
      abline(h = dat$betas[p.idx[j]], col = "red")
    }
  }

  if ("gamma" %in% estimator) {
    bvs.mcmc <- eval(parse(text = paste0("datMCMC$output$", estimator, "s")))
    #bvs.mcmc <- bvs.mcmc[, -1]
    p <- NCOL(dat$x)
    bvs.mcmc <- bvs.mcmc[, -seq(1, ((p + 1) * L), by = p + 1)]
    nIter <- nrow(bvs.mcmc)
    bvs.mcmc <- array(as.vector(bvs.mcmc), dim = c(nIter, p, L))

    layout(matrix(1:L, nrow = 1))
    par(mar = c(2, 4.1, 2, 2))

    for (l in 1:L) {
      plot(c(1:p) ~ 1,
        type = "n",
        xlim = c(1, nIter),
        yaxt = "n",
        ylab = "",
        xlab = "Iteration",
        main = paste0("Y", l)
      )
      axis(2, at = 1:p, labels = paste0("x", p:1), tick = FALSE, las = 1)
      for (j in 1:p) {
        for (i in 1:nIter) {
          #if (bvs.mcmc[i, p * (l - 1) + j] == 1) {
          if (bvs.mcmc[i, j, l] == 1) {
            segments(x0 = i - 0.4, y0 = p - j + 1, x1 = i + 0.4, y1 = p - j + 1, lwd = 1)
          } # Draw line
        }
      }
    }
  }

  if (any(estimator %in% c("kappa", "tau", "sigma"))) {
    layout(matrix(seq_along(estimator), ncol = 1))
    par(mar = c(2, 4.1, 2, 2))
    if ("kappa" %in% estimator) {
      kappa.mcmc <- datMCMC$output$kappa
      plot(kappa.mcmc,
        type = "l", lty = 1,
        ylab = expression(kappa), xlab = "MCMC iteration",
        ylim = summary(c(kappa.mcmc, dat$kappa))[c(1, 6)]
      )
      abline(h = dat$kappa, col = "red")
    }

    if ("tau" %in% estimator) {
      tauSq.mcmc <- datMCMC$output$tauSq
      plot(tauSq.mcmc,
        type = "l", lty = 1,
        ylab = expression(tau^2), xlab = "MCMC iteration",
        ylim = summary(as.vector(tauSq.mcmc))[c(1, 6)]
      )
    }
    
    if ("sigma" %in% estimator) {
      sigmaSq.mcmc <- datMCMC$output$sigmaSq
      plot(sigmaSq.mcmc,
           type = "l", lty = 1,
           ylab = expression(sigma^2), xlab = "MCMC iteration",
           ylim = summary(as.vector(sigmaSq.mcmc))[c(1, 6)]
      )
    }
  }
}
