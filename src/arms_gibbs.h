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

    static void arms_gibbs_beta_dirichlet(
        const armsParmClass& armsPar,
        const hyperparClass& hyperpar,
        arma::mat& currentPars,
        arma::umat gammas,
        arma::vec& tauSq,
        double& tau0Sq,
        const DataClass &dataclass
    );

    static void arms_gibbs_betaK_dirichlet(
        const unsigned int k,
        const armsParmClass& armsPar,
        const hyperparClass& hyperpar,
        arma::mat& currentPars,
        arma::umat gammas,
        double tauSqK,
        double tau0Sq,
        const DataClass &dataclass
    );

    static void arms_gibbs_beta_weibull(
        const armsParmClass& armsPar,
        const hyperparClass& hyperpar,
        arma::mat& currentPars,
        arma::umat gammas,
        double& tauSq,
        double& tau0Sq,

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

    static void slice_sample(
        double (*logfn)(double par, void *mydata),
        void *mydata,
        double& x,
        const unsigned int steps,
        const double w,
        const double lower,
        const double upper
    );

};


#endif
