<div style="text-align: left;">

### brmss 0.3 (2025-10-31) (GitHub Only)

* Fix minor bugs in `BVS_subfunc::gammaMC3Proposal()` and `BVS_gaussian::sampleGamma()
* Fix bug with Gibbs sampling for Bernoulli probability
* Add hierarchical related regressions (HRR)

* TODO: for (multivariate) probit model, try sample betas via ARMS

* TODO: add mvprobit
* TODO: define N, L, p as public (global) variables in each family class
* TODO: Check if better techniques instead of hard upper- or lower-bounds for exponential/log scales!
* TODO: add GLM for gaussian, poisson, bernoulli (logit, logistic, clog-log), gamma regression
* TODO: add logistic regression with Pólya-gamma augmentation
* TODO: add MRF prior

### brmss 0.2 (2025-10-29) (GitHub Only)

* Make C++ classes 'BVS_weibull' and 'BVS_dirichlet' for every model's MCMC for-loop, likelihood, gammaSample, which makes the code more generalizable
* Implement linear regression

### brmss 0.1 (2025-10-27) (GitHub Only)

* First draft version at `ocbe-uio` github

</div>