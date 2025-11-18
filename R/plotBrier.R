#' @title Plot curves of time-dependent Brier score
#'
#' @description
#' Predict time-dependent Brier scores based on different survival models
#'
#' @name plotBrier
#'
#' @importFrom survival Surv coxph 
#' @importFrom riskRegression Score
#' @importFrom stats median as.formula predict pweibull
#' @importFrom ggplot2 ggplot aes .data geom_step theme element_blank xlab ylab
#' @importFrom ggplot2 theme_bw guides guide_legend
#' @importFrom utils globalVariables
#' @importFrom graphics layout par abline
#'
#' @param dat input data as a list containing outcomes and covariates
#' @param datMCMC returned object from the main function \code{brmss()}
#' @param dat.new input data for out-sample prediction, with the same format
#' as \code{dat}
#' @param time.star largest time for survival prediction
#' @param xlab a title for the x axis
#' @param ylab a title for the y axis
#' @param ... other parameters
#'
#' @return A \code{ggplot2::ggplot} object. See \code{?ggplot2::ggplot} for more
#' details of the object.
#'
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
#' 
#' fit <- brmss(dat$y, dat$x, family = "weibull", nIter = 10, burnin = 1)
#' 
#' \donttest{
#' plotBrier(dat, datMCMC = fit)
#' }
#'
#' @export
plotBrier <- function(dat, datMCMC,
                      dat.new = NULL,
                      time.star = NULL,
                      xlab = "Time",
                      ylab = "Brier score", ...) {
  n <- NROW(dat$x)
  p <- NCOL(dat$x)
  if (datMCMC$input$family == "weibull") {
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

  # nIter <- datMCMC$input$nIter
  burnin <- datMCMC$input$burnin / datMCMC$input$thin

  # survival predictions based on posterior mean
  gammas.hat <- matrix(colMeans(datMCMC$output$gammas[-c(1:burnin), ]), ncol = L)
  #gammas.hat <- rbind(1, gammas.hat)
  betas.hat <- matrix(colMeans(datMCMC$output$betas[-c(1:burnin), ]), ncol = L)
  betas.hat <- (gammas.hat >= 0.5) * betas.hat / gammas.hat
  betas.hat[is.na(betas.hat)] <- 0

  if (datMCMC$input$family == "weibull") {
    kappa.hat <- mean(datMCMC$output$kappa[-c(1:burnin)])

    # predict survival probabilities based on brmss
    time_eval <- sort(survObj.new$time)
    Surv.prob <- matrix(nrow = n, ncol = length(time_eval))

    mu <- exp(cbind(1, dat.new$x) %*% betas.hat)
    lambdas <- mu / gamma(1 + 1 / kappa.hat)
    for (j in seq_along(time_eval)) {
      Surv.prob[, j] <- exp(-(time_eval[j] / lambdas)^kappa.hat)
    }
    pred.prob <- 1 - Surv.prob

    # other competing survival models
    if (p > 10) {
      message("Note: Classic survival models only use the first 10 covariates.")
    }
    x.names <- paste0("X", 1:min(p, 10))
    formula.tmp <- as.formula(paste0("Surv(time, event) ~ ", paste0(x.names, collapse = "+")))
    fitCox <- survival::coxph(formula.tmp, data = survObj, y = TRUE, x = TRUE)
    survfit0 <- survival::survfit(fitCox, survObj.new) # data.frame(x01=survObj.new$x01,x02=survObj.new$x02))
    pred.fitCox <- t(1 - summary(survfit0, times = time_eval, extend = TRUE)$surv)

    fitWeibull <- survival::survreg(formula.tmp, data = survObj, dist='weibull')#, scale=0, y = TRUE, x = TRUE)
    mu_hat <- predict(fitWeibull, newdata = data.frame(dat.new$x), type="link") 
    pred.fitWeibull <- t(sapply(mu_hat, function(i){
      pweibull(time_eval, shape = 1 / fitWeibull$scale, scale = exp(i))
      }))
    
    list.models <- list(
      "Cox" = pred.fitCox,
      "Weibull" = pred.fitWeibull,
      "brmss" = pred.prob
    )
    g <- riskRegression::Score(
      list.models,
      formula = Surv(time, event) ~ 1,
      metrics = "brier", summary = "ibs",
      data = survObj.new,
      conf.int = FALSE, times = time_eval
    )
    data <- g$Brier$score
    if (!is.null(time.star)) {
      data <- data[data$times <= time.star, ]
    }
    levels(data$model)[1] <- "Kaplan-Meier"
    g2 <- ggplot2::ggplot(data, aes(
      x = .data$times, y = .data$Brier, group = .data$model, color = .data$model
    )) +
      xlab(xlab) +
      ylab(ylab) +
      geom_step(direction = "vh") + # , alpha=0.4) +
      theme_bw() +
      guides(color = guide_legend(title = "Models"))
    # theme(
    #   legend.position = "inside",
    #   legend.position.inside = c(0.4, 0.25),
    #   legend.title = element_blank()
    # )
  }


  g2
}
