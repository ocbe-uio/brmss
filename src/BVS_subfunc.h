/* header file for Bayesian subfunctions*/

#ifndef BVS_SUBFUNC_H
#define BVS_SUBFUNC_H

#include "global.h"

class BVS_subfunc
{
public:

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

    static double logPDFBernoulli(unsigned int x, double pi);

    static double logPDFNormal(
        const arma::vec& x,
        double sigmaSq
    );

    static double logPDFNormal(
        const arma::vec& x,
        const arma::mat& Sigma
    );

    static double logPDFNormal(
        double x,
        double sigmaSq
    );

    static double logPDFNormal(
        const arma::vec& x,
        const arma::vec& m,
        const double Sigma
    );

    static double logPDFNormal(
        const arma::vec& x,
        const arma::vec& m,
        const arma::mat& Sigma
    );

    static arma::vec randMvNormal(
        const arma::vec &m,
        const arma::mat &Sigma
    );

    static double logPDFGamma(
        double x,
        double a,
        double b
    );

    static double logPDFIGamma(
        double x, double a, double b
    );

    static double randIGamma(
        double shape, double scale
    );

private:

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

    static arma::vec randVecNormal(
        const unsigned int n
    );


};


#endif
