/* header file for Bayesian variable selection via Metropolis-Hastings sampler*/

#ifndef BVS_iMVP_H
#define BVS_iMVP_H

#include "global.h"


class BVS_iMVP
{
public:

    static void mcmc(
        unsigned int nIter,
        unsigned int burnin,
        unsigned int thin,
        double& tau0Sq,
        arma::vec& tauSq,
        arma::mat& betas,
        arma::umat& gammas,
        const std::string& gammaProposal,
        Gamma_Sampler_Type gammaSampler,
        const armsParmClass& armsPar,
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

    static void loglikelihood_conditional(
        const arma::mat& betas,
        const arma::vec& sigmaSq,
        const DataClass &dataclass,
        arma::vec& loglik
    );

    static double loglikelihood(
        const arma::umat& gammas,
        const arma::vec& tauSq,
        double sigmaA,
        double sigmaB,
        const DataClass &dataclass
    );

    static void sampleGamma(
        arma::umat& gammas,
        Gamma_Sampler_Type gamma_sampler,
        arma::mat& logP_gamma,
        unsigned int& gamma_acc_count,
        arma::vec& loglik,

        const armsParmClass& armsPar,
        const hyperparClass& hyperpar,

        arma::mat& betas,
        double tau0Sq,
        arma::vec& tauSq,

        const DataClass &dataclass
    );

    static void sampleGammaProposalRatio(
        arma::umat& gammas,
        Gamma_Sampler_Type gamma_sampler,
        arma::mat& logP_gamma,
        unsigned int& gamma_acc_count,
        arma::vec& loglik,

        const armsParmClass& armsPar,
        const hyperparClass& hyperpar,

        arma::mat& betas,
        double tau0Sq,
        arma::vec& tauSq,

        const DataClass &dataclass
    );

    static double logPBeta(
        const arma::mat& betas,
        const arma::vec& tauSq,
        const DataClass& dataclass
    );

    static void gibbs_sigmaSq(
        arma::vec& sigmaSq,
        double a,
        double b,
        const DataClass& dataclass,
        const arma::mat& betas
    );

    static void gibbs_betas(
        arma::mat& betas,
        const arma::umat& gammas,
        const arma::vec& sigmaSq,
        const double tau0Sq,
        const arma::vec& tauSq,
        const DataClass &dataclass
    );

};


#endif
