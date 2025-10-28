/* header file for Bayesian variable selection via Metropolis-Hastings sampler*/

#ifndef BVS_GAUSSIAN_H
#define BVS_GAUSSIAN_H

#include "global.h"


class BVS_gaussian
{
public:

    static void mcmc(
        unsigned int nIter,
        unsigned int burnin,
        unsigned int thin,
        double sigmaSq,
        double& tau0Sq,
        arma::vec& tauSq,
        arma::mat& betas,
        arma::umat& gammas,
        const std::string& gamma_proposal,
        Gamma_Sampler_Type gammaSampler,
        const hyperparClass& hyperpar,
        const DataClass &dataclass,

        arma::vec& sigmaSq_mcmc,
        double& sigmaSq_post,
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
        double sigmaSq,
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
        double sigmaSq,
        double& tau0Sq,
        arma::vec& tauSq,

        const DataClass &dataclass
    );

    static void sampleGammaProposalRatio(
        arma::umat& gammas,
        Gamma_Sampler_Type gamma_sampler,
        arma::mat& logP_gamma,
        unsigned int& gamma_acc_count,
        arma::vec& loglik,

        const hyperparClass& hyperpar,

        arma::mat& betas,
        double sigmaSq,
        double& tau0Sq,
        arma::vec& tauSq,

        const DataClass &dataclass
    );

    static double gibbs_sigmaSq(
    double a,
    double b,
    const arma::vec& betas,
    const DataClass& dataclass
);

    static arma::vec randMvNormal(
        const arma::vec &m,
        const arma::mat &Sigma
    );

    static arma::vec randVecNormal(
        const unsigned int n
    );

    static double gibbs_beta_gaussian(
        arma::mat& betas,
        const arma::umat& gammas,
        double tau0Sq,
        double tauSq,
        double sigmaSq,
        const DataClass& dataclass
    );

    static double logPbeta(
        const arma::mat& betas,
        double tau0Sq,
        double tauSq,
        double sigmaSq,
        const DataClass& dataclass
    );

    static double logPDFNormal(
        const arma::vec& x,
        const arma::vec& m,
        const arma::mat& Sigma
    );
};


#endif
