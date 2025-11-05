/* header file for updating variances using classical Gibbs sampler */

#ifndef SIMPLE_GIBBS_H
#define SIMPLE_GIBBS_H

#include <cmath>
#include <RcppArmadillo.h>


double sampleTau0(
    double a,
    double b,
    double x
);

double sampleTau(
    double a,
    double b,
    const arma::mat& betas
);

#endif
