/* header file for Bayesian variable selection via Metropolis-Hastings sampler*/

#ifndef BVS_WEIBULL_H
#define BVS_WEIBULL_H

#include "global.h"


class BVS_weibull
{
public:

    static void mcmc(
        unsigned int nIter,
        unsigned int burnin,
        unsigned int thin,
        double kappa,
        double& tau0Sq,
        arma::vec& tauSq,
        arma::mat& betas,
        arma::umat& gammas,
        const std::string& gammaProposal,
        Gamma_Sampler_Type gammaSampler,
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


private:

    // log-density of survival and measurement error data
    static void loglikelihood(
        const arma::mat& betas,
        double kappa,
        const DataClass &dataclass,
        arma::vec& loglik
    );

    static void sampleGamma(
        arma::umat& gammas,
        Gamma_Sampler_Type gamma_sampler,
        const std::string& gammaProposal,
        arma::mat& logP_gamma,
        unsigned int& gamma_acc_count,
        arma::vec& loglik,

        const armsParmClass& armsPar,
        const hyperparClass& hyperpar,

        arma::mat& betas,
        double kappa,
        double tau0Sq,
        arma::vec& tauSq,

        const DataClass &dataclass
    );

    static double logPBeta(
        const arma::mat& betas,
        double tauSq,
        double kappa,
        const DataClass& dataclass
    );
};


#endif
