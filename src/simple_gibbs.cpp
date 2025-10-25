// Gibbs sampling for variance parameters

#include "simple_gibbs.h"
#include <stdio.h>


// update \beta0's variance tau0Sq
double sampleTau0(
    double a,
    double b,
    double x
)
{
    a += 0.5;
    b += 0.5 * x * x;

    return ( 1. / R::rgamma(a, 1. / b) );
}

// update \betas' variance tauSq
double sampleTau(
    double a,
    double b,
    const arma::vec& betas
)
{
    a += 0.5 * arma::sum(betas != 0.);
    b += 0.5 * arma::sum(betas % betas);

    return ( 1. / R::rgamma(a, 1. / b) );
}
