<div style="text-align: left;">

### brmss 0.4 (2025-11-27) (GitHub Only)

* Suppress `gammaProposal == "simple"` in `BVS_logistic.cpp` and use RW-MH for proposing betas in the gamma-beta move
* Add argument `RW.MH = c("fisher", "adaptive", "symmetric")` in `brmss()` for choosing the type fo RW's variance
* Implement `RW.MH = "symmetric"` in `BVS_weibull.cpp`

* TODO: in MVP models, try to use MH sampler for D instead of directly extracting from Psi

* TODO: drop beta_post and gamma_post to save memory

* TODO: add mvprobit
* TODO: define N, L, p as public (global) variables in each family class
* TODO: Check if better techniques instead of hard upper- or lower-bounds for exponential/log scales!
* TODO: add GLM for gaussian, poisson, bernoulli (logit, logistic, clog-log), gamma regression
* TODO: add logistic regression with Pólya-gamma augmentation
* TODO: add MRF prior
* TODO: pass 'logP_beta' in gammaSample() to avoid repeatedly update

### brmss 0.3 (2025-10-31) (GitHub Only)

* Fix minor bugs in `BVS_subfunc::gammaMC3Proposal()` and `BVS_gaussian::sampleGamma()
* Fix bug with Gibbs sampling for Bernoulli probability
* Add hierarchical related regressions (HRR)
* Fixed bugs in all functions `sampleGammaProposalRatio()`

### brmss 0.2 (2025-10-29) (GitHub Only)

* Make C++ classes 'BVS_weibull' and 'BVS_dirichlet' for every model's MCMC for-loop, likelihood, gammaSample, which makes the code more generalizable
* Implement linear regression

### brmss 0.1 (2025-10-27) (GitHub Only)

* First draft version at `ocbe-uio` github

</div>