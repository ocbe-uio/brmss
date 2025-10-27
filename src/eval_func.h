/* header file for evaluation functions (i.e. log densities) */

#ifndef EVAL_FUNC_H
#define EVAL_FUNC_H

#include <stdio.h>
#include <RcppArmadillo.h>


// typedef std::vector<double> stdvec;


typedef struct common_data
{
    // members
    double *currentPars;

    unsigned int jj;
    unsigned int l;
    unsigned int p;
    unsigned int L;
    unsigned int N;

    double tau0Sq;
    double tauSq;

    double kappa;
    double kappaA;
    double kappaB;
    double *mu;
    double *logMu;
    const double *X;
    const double *y;
    const unsigned int *event;
} dataS;

class EvalFunction
{
public:


    static double log_dens_betas_dirichlet
    (
        double par,
        void *abc_data
    );

    static double log_dens_betas_weibull
    (
        double par,
        void *abc_data
    );

    static double log_dens_kappa
    (
        double par,
        void *abc_data
    );

};

#endif
