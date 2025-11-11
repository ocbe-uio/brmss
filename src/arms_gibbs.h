/* header file for univariate and multivariate arms for all parameters */

#ifndef ARMS_GIBBS_H
#define ARMS_GIBBS_H

#include <cmath>

#include "arms.h"
#include "eval_func.h"
#include "global.h"


class ARMS_Gibbs
{
public:

    static void arms_gibbs_beta_logistic(
        const armsParmClass& armsPar,
        const hyperparClass& hyperpar,
        arma::mat& currentPars,
        arma::umat gammas,
        double tau0Sq,
        double tauSq,
        const DataClass &dataclass
    );

    static void arms_gibbs_beta_dirichlet(
        const armsParmClass& armsPar,
        const hyperparClass& hyperpar,
        arma::mat& currentPars,
        arma::umat gammas,
        double tau0Sq,
        arma::vec& tauSq,
        const DataClass &dataclass
    );

    static void arms_gibbs_betaK_dirichlet(
        const unsigned int k,
        const armsParmClass& armsPar,
        const hyperparClass& hyperpar,
        arma::mat& currentPars,
        arma::umat gammas,
        double tau0Sq,
        double tauSqK,
        const DataClass &dataclass
    );

    static void arms_gibbs_beta_weibull(
        const armsParmClass& armsPar,
        const hyperparClass& hyperpar,
        arma::mat& currentPars,
        arma::umat gammas,
        double tau0Sq,
        double tauSq,

        double kappa,
        const DataClass &dataclass
    );

    static void slice_kappa(
        double& currentPars,
        double minD,
        double maxD,
        double kappaA,
        double kappaB,
        const DataClass &dataclass,
        arma::vec& logMu
    );

    static double slice_sample(
        double (*logfn)(double par, void *mydata),
        void *mydata,
        double x,
        const unsigned int steps,
        const double w,
        const double lower,
        const double upper
    );

};


#endif
