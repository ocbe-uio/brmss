// Main function implemented the MCMC loop for weibull model
#ifndef WEIBULL_H
#define WEIBULL_H

#include "simple_gibbs.h"
#include "arms_gibbs.h"
#include "BVS.h"
#include "global.h"

void weibull(
    unsigned int nIter,
    unsigned int burnin,
    unsigned int thin,
    double kappa,
    double& tau0Sq,
    double& tauSq,
    arma::mat& betas,
    arma::umat& gammas,
    const std::string& gamma_proposal,
    Gamma_Sampler_Type gammaSampler,
    Family_Type familyType,
    const armsParmClass& armsPar,
    const hyperparClass& hyperpar,
    const DataClass &dataclass,

    arma::vec& kappa_mcmc,
    double& kappa_post,
    arma::mat& beta_mcmc,
    arma::mat& beta_post,
    arma::umat& gamma_mcmc,
    arma::umat& gamma_post,
    unsigned int& gamma_acc_count,
    arma::mat& loglikelihood_mcmc,
    arma::vec& tauSq_mcmc
);

#endif
