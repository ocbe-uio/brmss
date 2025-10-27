// Main function implemented the MCMC loop for weibull model


#include "simple_gibbs.h"
#include "arms_gibbs.h"
#include "BVS.h"
#include "global.h"
#include "weibull.h"

void weibull(
    unsigned int nIter,
    unsigned int burnin,
    unsigned int thin,
    double kappa,
    double& tau0Sq,
    double& tauSq,
    arma::mat& betas,
    arma::umat& gammas,
    const std::string& gamma_proposal,
    Gamma_Sampler_Type gammaSampler,
    Family_Type familyType,
    const armsParmClass& armsPar,
    const hyperparClass& hyperpar,
    const DataClass &dataclass,

    arma::vec& kappa_mcmc,
    double& kappa_post,
    arma::mat& beta_mcmc,
    arma::mat& beta_post,
    arma::umat& gamma_mcmc,
    arma::umat& gamma_post,
    unsigned int& gamma_acc_count,
    arma::mat& loglikelihood_mcmc,
    arma::vec& tauSq_mcmc
)
{
    // dimensions
    unsigned int N = dataclass.X.n_rows;
    unsigned int p = dataclass.X.n_cols;
    unsigned int L = dataclass.y.n_cols;

    arma::mat logP_gamma = arma::zeros<arma::mat>(p, L);; // this is declared to be updated in the M-H sampler for gammas

    gamma_acc_count = 0;
    for(unsigned int l=0; l<L; ++l)
    {
        double pi = R::rbeta(hyperpar.piA, hyperpar.piB);

        for(unsigned int j=0; j<p; ++j)
        {
            gammas(j, l) = R::rbinom(1, pi);
            logP_gamma(j, l) = BVS_Sampler::logPDFBernoulli(gammas(j, l), pi);
        }
    }

    arma::vec loglik = arma::zeros<arma::vec>(N);
    BVS_Sampler::loglikelihood(
        betas,
        kappa,
        dataclass,
        loglik
    );
    loglikelihood_mcmc.row(0) = loglik.t();

    // mean parameter
    arma::mat mu = arma::zeros<arma::mat>(N, L);
    arma::vec lambdas; // Weibull's scale parameter

    arma::vec logMu = arma::zeros<arma::mat>(N, L);
    logMu = betas(0) + dataclass.X * betas.submat(1, 0, p, 0);

    BVS_Sampler::loglikelihood(
        betas,
        kappa,
        dataclass,
        loglik
    );
    loglikelihood_mcmc.row(0) = loglik.t();


    // ###########################################################
    // ## MCMC loop
    // ###########################################################

    const unsigned int cTotalLength = 50;
    //std::cout
    Rprintf("Running MCMC iterations ...\n");
    unsigned int nIter_thin_count = 0;
    for (unsigned int m=0; m<nIter; ++m)
    {
        // print progression cursor
        if (m % 10 == 0 || m == (nIter - 1))
            //std::cout
            Rcpp::Rcout << "\r[" <<                                           //'\r' aka carriage return should move printer's cursor back at the beginning of the current line
                        std::string(cTotalLength * (m + 1.) / nIter, '#') <<        // printing filled part
                        std::string(cTotalLength * (1. - (m + 1.) / nIter), '-') << // printing empty part
                        "] " << (int)((m + 1.) / nIter * 100.0) << "%\r";             // printing percentage


        // update Weibull's shape parameter kappa
        ARMS_Gibbs::slice_kappa(
            kappa,
            armsPar.kappaMin,
            armsPar.kappaMax,
            hyperpar.kappaA,
            hyperpar.kappaB,
            dataclass,
            logMu
        );

        // update Weibull's quantities based on the new kappa
        // lambdas = mu.col(0) / std::tgamma(1.0+1.0/kappa);
        // weibullS = arma::exp(- arma::pow( y/lambdas, kappa));

        // update \gammas -- variable selection indicators
        if (gamma_proposal == "simple")
        {
            BVS_Sampler::sampleGamma(
                gammas,
                gammaSampler,
                familyType,
                logP_gamma,
                gamma_acc_count,
                loglik,
                armsPar,
                hyperpar,
                betas,
                kappa,
                tau0Sq,
                tauSq,

                dataclass
            );
        }
        else
        {
            BVS_Sampler::sampleGammaProposalRatio(
                gammas,
                gammaSampler,
                familyType,
                logP_gamma,
                gamma_acc_count,
                loglik,
                armsPar,
                hyperpar,
                betas,
                kappa,
                tau0Sq,
                tauSq,

                dataclass
            );
        }

        // update \betas
        ARMS_Gibbs::arms_gibbs_beta_weibull(
            armsPar,
            hyperpar,
            betas,
            gammas,
            tauSq,
            tau0Sq,

            kappa,
            dataclass
        );

        // update \betas' variance tauSq
        // hyperpar.tauSq = sampleTau(hyperpar.tauA, hyperpar.tauB, betas);
        // tauSq_mcmc[1+m] = hyperpar.tauSq;

#ifdef _OPENMP
        #pragma omp parallel for
#endif

        // update Weibull's quantities based on the new betas
        for(unsigned int l=0; l<L; ++l)
        {
            logMu = betas(0) + dataclass.X * betas.submat(1, l, p, l);
            // logMu.elem(arma::find(logMu > upperbound)).fill(upperbound);
            // mu.col(l) = arma::exp( logMu );
        }

        // save results for un-thinned posterior mean
        if(m >= burnin)
        {
            kappa_post += kappa;
            beta_post += betas;
            gamma_post += gammas;
        }

        // save results of thinned iterations
        if((m+1) % thin == 0)
        {
            kappa_mcmc[1+nIter_thin_count] = kappa;
            beta_mcmc.row(1+nIter_thin_count) = arma::vectorise(betas).t();
            tauSq_mcmc[1+nIter_thin_count] = tauSq;//hyperpar.tauSq; // TODO: only keep the firs one for now

            gamma_mcmc.row(1+nIter_thin_count) = arma::vectorise(gammas).t();

            BVS_Sampler::loglikelihood(
                betas,
                kappa,
                dataclass,
                loglik
            );

            // save loglikelihoods
            loglikelihood_mcmc.row(1+nIter_thin_count) = loglik.t();

            ++nIter_thin_count;
        }

    }

}

