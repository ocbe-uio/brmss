/* header file for global variables*/

#ifndef GLOBAL_H
#define GLOBAL_H

#include <random>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <RcppArmadillo.h>

// Define constants for bounds
// constexpr double UPPER_BOUND = 700.0;
constexpr double UPPER_BOUND_2 = 1.0e10; // 1.0e100 will result in numeric issue in ARMS algorithm for Weibull model
// constexpr double UPPER_BOUND_3 = 170.0;
// constexpr double LOWER_BOUND = 1.0e-10;
constexpr double UPPER_BOUND_3 = 1.0e100;
constexpr double LOWER_BOUND = 1.0e-100;

// Using the constants inline where necessary
// inline double upperbound = UPPER_BOUND;
inline double upperbound2 = UPPER_BOUND_2;
inline double upperbound3 = UPPER_BOUND_3;
inline double lowerbound = LOWER_BOUND;


enum class Family_Type
{
    gaussian=1, logit, probit, poisson, beta, weibull, dirichlet
}; // scoped enum

enum class Gamma_Sampler_Type
{
    bandit = 1, mc3, gibbs
}; // scoped enum

class armsParmClass
{
public:
    // members
    const unsigned int n;
    const int nsamp;
    const int ninit;
    const int metropolis;
    const double convex;
    const int npoint;

    // ranges of key parameters
    const double kappaMin;
    const double kappaMax;
    const double betaMin;
    const double betaMax;

    // Constructor to initialize the constants
    armsParmClass(
        unsigned int n_,
        int nsamp_,
        int ninit_,
        int metropolis_,
        double convex_,
        int npoint_,

        double kappaMin_,
        double kappaMax_,
        double betaMin_,
        double betaMax_
    ) :
        n(n_),
        nsamp(nsamp_),
        ninit(ninit_),
        metropolis(metropolis_),
        convex(convex_),
        npoint(npoint_),

        kappaMin(kappaMin_),
        kappaMax(kappaMax_),
        betaMin(betaMin_),
        betaMax(betaMax_)
    {}

} ;


class hyperparClass
{
public:
    // members
    const double piA;
    const double piB;

    const double sigmaA;
    const double sigmaB;

    // const double tau0Sq;
    const double tau0A;
    const double tau0B;
    // const double tauSq;
    const double tauA;
    const double tauB;

    const double kappaA;
    const double kappaB;

    // Constructor to initialize the constants
    hyperparClass(
        double piA_,
        double piB_,

        double sigmaA_,
        double sigmaB_,

        double tau0A_,
        double tau0B_,
        double tauA_,
        double tauB_,

        double kappaA_,
        double kappaB_
    ) :
        piA(piA_),
        piB(piB_),

        sigmaA(sigmaA_),
        sigmaB(sigmaB_),

        tau0A(tau0A_),
        tau0B(tau0B_),
        tauA(tauA_),
        tauB(tauB_),

        kappaA(kappaA_),
        kappaB(kappaB_)
    {}
} ;

class DataClass
{
public:
    // Use const for immutable members
    const arma::mat X;
    const arma::mat y;
    const arma::uvec event;

    // Constructor to initialize the constants
    DataClass(
        const arma::mat& X_,
        const arma::mat& y_,
        const arma::uvec& event_
    ) :
        X(X_),
        y(y_),
        event(event_)
    {}
} ;

#endif
