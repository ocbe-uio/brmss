/* header file for Bayesian variable selection via Metropolis-Hastings sampler*/

#ifndef BVS_LOGISTIC_H
#define BVS_LOGISTIC_H

#include "global.h"


class BVS_logistic
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
        const std::string& rw_mh,
        const armsParmClass& armsPar,
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
    static arma::vec loglikelihood(
        const arma::mat& betas,
        const DataClass &dataclass
    );

    static void sampleGamma(
        arma::umat& gammas,
        Gamma_Sampler_Type gamma_sampler,
        const std::string& gammaProposal,
        const std::string& rw_mh,
        arma::mat& logP_gamma,
        unsigned int& gamma_acc_count,
        double& logP_beta,
        arma::vec& loglik,

        const armsParmClass& armsPar,
        const hyperparClass& hyperpar,

        arma::mat& betas,
        double tau0Sq,
        double tauSq,
        const unsigned int iter,
        const unsigned int burnin,

        const DataClass &dataclass
    );

    static arma::mat calculateLambda(
        const arma::mat& betas,
        const double tauSq,
        const arma::uvec& updateIdx,
        const DataClass& dataclass
    );
    /*
    static double logPBeta(
        const arma::mat& betas,
        double tauSq,
        const DataClass& dataclass
    );
    */
};


#endif
