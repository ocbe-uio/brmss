/* header file for Bayesian variable selection via Metropolis-Hastings sampler*/

#ifndef BVS_PROBIT_H
#define BVS_PROBIT_H

#include "global.h"


class BVS_probit
{
public:

    static void mcmc(
        unsigned int nIter,
        unsigned int burnin,
        unsigned int thin,
        double& tau0Sq_,
        arma::vec& tauSq_,
        arma::mat& betas,
        arma::umat& gammas,
        const std::string& gammaProposal,
        Gamma_Sampler_Type gammaSampler,
        const hyperparClass& hyperpar,
        const DataClass &dataclass,

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
        const DataClass &dataclass,
        arma::vec& loglik
    );

    static void sampleGamma(
        arma::umat& gammas,
        Gamma_Sampler_Type gamma_sampler,
        arma::mat& logP_gamma,
        unsigned int& gamma_acc_count,
        arma::vec& loglik,

        const hyperparClass& hyperpar,

        arma::mat& betas,
        const arma::vec& z,
        double tau0Sq,
        double tauSq,

        const DataClass &dataclass
    );

    static void sampleGammaProposalRatio(
        arma::umat& gammas,
        Gamma_Sampler_Type gamma_sampler,
        arma::mat& logP_gamma,
        unsigned int& gamma_acc_count,
        double& logP_beta,
        arma::vec& loglik,

        const hyperparClass& hyperpar,

        arma::mat& betas,
        const arma::vec& z,
        double tau0Sq,
        double tauSq,

        const DataClass &dataclass
    );

    static double gibbs_beta_probit(
        arma::mat& betas,
        const arma::umat& gammas,
        double tau0Sq,
        double tauSq,
        const arma::vec& z,
        const DataClass& dataclass
    );

    static arma::vec zbinprobit(
        const arma::mat& betas,
        const DataClass& dataclass
    );

};


#endif
