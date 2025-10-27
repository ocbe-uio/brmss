/* header file for Bayesian variable selection via Metropolis-Hastings sampler*/

#ifndef BVS_H
#define BVS_H

#include "global.h"

#include <RcppArmadillo.h>

enum class Family_Type
{
    weibull = 1, dirichlet
}; // scoped enum

enum class Gamma_Sampler_Type
{
    bandit = 1, mc3
}; // scoped enum

/*
typedef struct HyperparData
{
    // members
    double piA;
    double piB;

    double tau0A;
    double tau0B;
    double tauA;
    double tauB;

    double kappaA;
    double kappaB;
} hyperparS;
*/

class BVS_Sampler
{
public:
    BVS_Sampler(
        const DataClass& dataclass
    ) :
        dataclass_(dataclass) {}

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
        Family_Type familyType,
        arma::mat& logP_gamma,
        unsigned int& gamma_acc_count,
        arma::vec& loglik,

        const armsParmClass& armsPar,
        const hyperparClass& hyperpar,

        arma::mat& betas,
        double kappa,
        double& tau0Sq,
        double& tauSq,

        const DataClass &dataclass
    );

    static void sampleGammaProposalRatio(
        arma::umat& gammas,
        Gamma_Sampler_Type gamma_sampler,
        Family_Type familyType,
        arma::mat& logP_gamma,
        unsigned int& gamma_acc_count,
        arma::vec& loglik,

        const armsParmClass& armsPar,
        const hyperparClass& hyperpar,

        arma::mat& betas,
        double kappa,
        double& tau0Sq,
        double& tauSq,

        const DataClass &dataclass
    );

    static double logPDFBernoulli(unsigned int x, double pi);

private:
    DataClass dataclass_;

    static double gammaMC3Proposal(
        unsigned int p,
        arma::umat& mutantGammas,
        const arma::umat gammas,
        arma::uvec& updateIdx,
        unsigned int componentUpdateIdx_
    );

    static double gammaBanditProposal(
        unsigned int p,
        arma::umat& mutantGammas,
        const arma::umat gammas,
        arma::uvec& updateIdx,
        unsigned int componentUpdateIdx,
        arma::mat& banditAlpha
    );

    static arma::uvec randWeightedIndexSampleWithoutReplacement(
        unsigned int populationSize,
        const arma::vec& weights,
        unsigned int sampleSize
    );

    static unsigned int randWeightedIndexSampleWithoutReplacement(
        const arma::vec& weights
    );

    static double logPDFWeightedIndexSampleWithoutReplacement(
        const arma::vec& weights,
        const arma::uvec& indexes
    );

    static double logspace_add(
        double a,
        double b
    );

    static double logPDFNormal(
        const arma::vec& x,
        const double& sigmaSq
    );

    static double logPbeta(
        const arma::mat& betas,
        double tauSq,
        double kappa,
        const DataClass& dataclass
    );

};


#endif
