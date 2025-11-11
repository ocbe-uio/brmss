/* header file for Bayesian variable selection via Metropolis-Hastings sampler*/

#ifndef BVS_dMVP_H
#define BVS_dMVP_H

#include "global.h"


class BVS_dMVP
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

    static void loglikelihood_conditional(
        const arma::mat& betas,
        const arma::mat& D,
        const DataClass &dataclass,
        arma::vec& loglik
    );

    static double logLikelihood(
        const arma::mat& betas,
        const arma::mat& D,
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
        const double tau0Sq,
        const double tauSq,
        const arma::mat& SigmaRho,
        const arma::mat& RhoU,

        const arma::mat& Z,
        const arma::mat& D,
        const DataClass &dataclass
    );

    static void sampleGammaProposalRatio(
        arma::umat& gammas,
        Gamma_Sampler_Type gamma_sampler,
        arma::mat& logP_gamma,
        unsigned int& gamma_acc_count,
        double& log_likelihood,
        const hyperparClass& hyperpar,

        arma::mat& betas,
        const double tau0Sq,
        const double tauSq,
        const arma::mat& SigmaRho,
        const arma::mat& RhoU,

        const arma::mat& Z,
        const arma::mat& D,
        const DataClass &dataclass
    );

    static void gibbs_SigmaRho(
        arma::mat& SigmaRho,
        // const double psi,
        arma::mat& RhoU,
        const double nu,
        double& logP_SigmaRho,
        const arma::mat& Z,
        const DataClass& dataclass,
        const arma::mat& betas
    );
    /*
    static double logPSigmaRho(
        const arma::mat& SigmaRho,
        const unsigned int N,
        const double nu
    );

    static void samplePsi(
        double& psi,
        const double psiA,
        const double psiB,
        const double nu,
        double& logP_psi,
        double& logP_SigmaRho,
        const arma::mat& SigmaRho
    );
    */

    static arma::mat createRhoU(
        const arma::mat& U,
        const arma::mat&  SigmaRho
    );

    static void gibbs_betas(
        arma::mat& betas,
        const arma::umat& gammas,
        const arma::mat& SigmaRho,
        const arma::mat& RhoU,
        const double tau0Sq,
        const double tauSq,
        const arma::mat& Z,
        const DataClass &dataclass
    );

    static double logP_gibbs_betaK(
        const unsigned int k,
        const arma::mat& betas,
        const arma::umat& gammas,
        const arma::mat& SigmaRho,
        arma::mat& RhoU,
        const double tau0Sq,
        const double tauSq,
        const arma::mat& Z,
        const DataClass &dataclass
    );

    static double logPBetaMask(
        const arma::mat& betas,
        const arma::umat& gammas,
        const double tau0Sq,
        const double tauSq
    );

    static double gibbs_betaK(
        const unsigned int k,
        arma::mat& betas,
        const arma::umat& gammas,
        const arma::mat& SigmaRho,
        arma::mat& RhoU,
        const double tau0Sq,
        const double tauSq,
        const arma::mat& Z,
        const DataClass &dataclass
    );

    static void sampleZ(
        arma::mat& mutantZ,
        arma::mat& mutantD,
        const arma::mat& betas,
        const arma::mat& Psi,
        const DataClass &dataclass
    );

    static arma::vec zbinprobit(
        const arma::vec& y,
        const arma::vec& m
    );

    static double zbinprobit(
        const double y,
        const double m,
        const double sigma
    );

    static void updatePsi(
        const arma::mat& SigmaRho,
        arma::mat& Psi
    );

};


#endif
