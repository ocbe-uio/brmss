/* Evaluation functions (i.e. log densities) for ARS and ARMS*/

#include <memory> // Include for smart pointers

#include "eval_func.h"
#include "global.h"


// log-density for coefficient betas
double EvalFunction::log_dens_betas_weibull(
    double par,
    void *abc_data)
{
    double h = 0.;

    // Allocation of a zero-initialized memory block of (num*size) bytes
    auto mydata_parm = static_cast<dataS*>(abc_data);

    arma::uvec event(const_cast<unsigned int*>(mydata_parm->event), mydata_parm->N, false);
    arma::mat y(const_cast<double*>(mydata_parm->y), mydata_parm->N, 1, false);
    arma::mat X(const_cast<double*>(mydata_parm->X), mydata_parm->N, mydata_parm->p, false);

    arma::mat pars(mydata_parm->currentPars, mydata_parm->p+1, 1, false);
    pars(mydata_parm->jj) = par;

    arma::vec logMu = pars(0) + X * pars.submat(1, 0, mydata_parm->p, 0);
    // logMu.elem(arma::find(logMu > upperbound)).fill(upperbound);

    arma::vec lambdas = arma::exp(logMu) / std::tgamma(1. + 1./mydata_parm->kappa);

    double tau = mydata_parm->tauSq;
    if(mydata_parm->jj == 0)
    {
        tau = mydata_parm->tau0Sq;
    }
    double logprior = - par * par / tau / 2.;

    arma::vec logpost_first = std::log(mydata_parm->kappa) -
                              mydata_parm->kappa * (logMu - std::lgamma(1. + 1./mydata_parm->kappa)) + //arma::log(lambdas) +
                              (mydata_parm->kappa - 1) * arma::log(y);
    double logpost_first_sum = arma::sum( logpost_first.elem(arma::find(event == 1)) );

    arma::vec logpost_second = arma::pow( y / lambdas, mydata_parm->kappa);
    logpost_second.elem(arma::find(logpost_second > upperbound9)).fill(upperbound9);
    double logpost_second_sum =  arma::sum( - logpost_second );

    h = logpost_first_sum + logpost_second_sum + logprior;

    // free(mydata_parm);
    return h;
}


// log-density for kappa
double EvalFunction::log_dens_kappa(
    double par,
    void *abc_data)
{
    double h = 0.;

    // dataS *mydata_parm = (dataS *)calloc(sizeof(dataS), sizeof(dataS));
    // std::unique_ptr<dataS> mydata_parm = std::make_unique<dataS>();
    // *mydata_parm = *(dataS *)abc_data;
    auto mydata_parm = static_cast<dataS*>(abc_data);

    double logprior = R::dgamma(par, mydata_parm->kappaA, 1.0/mydata_parm->kappaA, true);

    arma::vec logMu(mydata_parm->logMu, mydata_parm->N, 1, false);
    arma::mat y(const_cast<double*>(mydata_parm->y), mydata_parm->N, 1, false);
    arma::uvec event(const_cast<unsigned int*>(mydata_parm->event), mydata_parm->N, false);

    arma::vec lambdas = arma::exp(logMu) / std::tgamma(1.0+1.0/par);

    arma::vec logpost_first = std::log(par) - par * (logMu - std::lgamma(1. + 1./par)) + //arma::log(lambdas) +
                              (par - 1) * arma::log(y);
    double logpost_first_sum = arma::sum( logpost_first.elem(arma::find(event == 1)) );
    double logpost_second_sum =  arma::sum( - arma::pow( y / lambdas, par) );


    h = logpost_first_sum + logpost_second_sum + logprior;

    // std::cout << "...debug log_dens_kappa h=" << h << "\n";
    // free(mydata_parm);
    return h;
}
