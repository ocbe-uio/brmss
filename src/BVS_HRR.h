/* header file for Bayesian variable selection via Metropolis-Hastings sampler*/

#ifndef BVS_HRR_H
#define BVS_HRR_H

#include "global.h"


class BVS_HRR
{
public:

    static void mcmc(
        unsigned int nIter,
        unsigned int burnin,
        unsigned int thin,
        arma::mat& betas,
        arma::umat& gammas,
        const std::string& gammaProposal,
        Gamma_Sampler_Type gammaSampler,
        const hyperparClass& hyperpar,
        const DataClass &dataclass,

        arma::vec& sigmaSq_mcmc,
        arma::mat& beta_mcmc,
        arma::mat& beta_post,
        arma::umat& gamma_mcmc,
        arma::umat& gamma_post,
        unsigned int& gamma_acc_count,
        arma::mat& loglikelihood_mcmc,
        arma::vec& tauSq_mcmc
    );


private:

    static void sampleTau(
        double& tauSq,
        double& logP_tau,
        double& log_likelihood,
        const hyperparClass& hyperpar,
        const DataClass& dataclass,
        const arma::umat& gammas
    );

    static void loglikelihood_conditional(
        const arma::mat& betas,
        const arma::vec& sigmaSq,
        const DataClass &dataclass,
        arma::vec& loglik
    );

    static double logLikelihood(
        const arma::umat& gammas,
        const double tauSq,
        const hyperparClass& hyperpar,
        const DataClass &dataclass
    );

    static void sampleGamma(
        arma::umat& gammas,
        Gamma_Sampler_Type gamma_sampler,
        arma::mat& logP_gamma,
        unsigned int& gamma_acc_count,
        double& log_likelihood,

        const hyperparClass& hyperpar,

        arma::mat& betas,
        const double tauSq,

        const DataClass &dataclass
    );

    static void gibbs_sigmaSq(
        arma::vec& sigmaSq,
        const double tauSq,
        const hyperparClass& hyperpar,
        const DataClass& dataclass,
        const arma::mat& betas
    );

    static void gibbs_betas(
        arma::mat& betas,
        const arma::umat& gammas,
        const arma::vec& sigmaSq,
        const double tauSq,
        const DataClass &dataclass
    );

};


#endif
