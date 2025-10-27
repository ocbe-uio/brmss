// Main function implemented the MCMC loop for dirichlet model

#include "simple_gibbs.h"
#include "arms_gibbs.h"
#include "BVS.h"
#include "global.h"
#include "dirichlet.h"

void dirichlet(
    unsigned int nIter,
    unsigned int burnin,
    unsigned int thin,
    double& tau0Sq,
    arma::vec& tauSq,
    arma::mat& betas,
    arma::umat& gammas,
    const std::string& gamma_proposal,
    Gamma_Sampler_Type gammaSampler,
    Family_Type familyType,
    const armsParmClass& armsPar,
    const hyperparClass& hyperpar,
    const DataClass &dataclass,

    arma::mat& beta_mcmc,
    arma::mat& beta_post,
    arma::umat& gamma_mcmc,
    arma::umat& gamma_post,
    unsigned int& gamma_acc_count,
    arma::mat& loglikelihood_mcmc,
    arma::vec& tauSq_mcmc
)
{
    // std::cout << "...debug11\n";
    // dimensions
    unsigned int N = dataclass.X.n_rows;
    unsigned int p = dataclass.X.n_cols;
    unsigned int L = dataclass.y.n_cols;

    double kappa = 0.; // TODO: defined for the general function 'sampleGamma()'

    arma::mat logP_gamma = arma::zeros<arma::mat>(p, L);; // this is declared to be updated in the M-H sampler for gammas
    
    // std::cout << "...debug12\n";
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
    // std::cout << "...debug13\n";
    arma::vec loglik = arma::zeros<arma::vec>(N);
    BVS_Sampler::loglikelihood(
        betas,
        kappa,
        dataclass,
        loglik
    );
    loglikelihood_mcmc.row(0) = loglik.t();
    // std::cout << "...debug14\n";
    arma::mat proportion;
    arma::mat alphas = arma::zeros<arma::mat>(N, L);
    for(unsigned int l=0; l<L; ++l)
    {
        alphas.col(l) = arma::exp( betas(0, l) + dataclass.X * betas.submat(1, l, p, l) );
    }
    alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
    alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
    proportion = alphas / arma::repmat(arma::sum(alphas, 1), 1, L);
    // std::cout << "...debug15\n";

    // ###########################################################
    // ## MCMC loop
    // ###########################################################

    const unsigned int cTotalLength = 50;

    Rprintf("Running MCMC iterations ...\n");
    unsigned int nIter_thin_count = 0;
    for (unsigned int m=0; m<nIter; ++m)
    {
        // print progression cursor
        if (m % 10 == 0 || m == (nIter - 1))
            Rcpp::Rcout << "\r[" <<                                           //'\r' aka carriage return should move printer's cursor back at the beginning of the current line
                        std::string(cTotalLength * (m + 1.) / nIter, '#') <<        // printing filled part
                        std::string(cTotalLength * (1. - (m + 1.) / nIter), '-') << // printing empty part
                        "] " << (int)((m + 1.) / nIter * 100.0) << "%\r";             // printing percentage

    // std::cout << "...debug16\n";
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
    // std::cout << "...debug17\n";
        // update all betas
        ARMS_Gibbs::arms_gibbs_beta_dirichlet(
            armsPar,
            hyperpar,
            betas,
            gammas,
            tauSq,
            tau0Sq,
            dataclass
        );
    // std::cout << "...debug18\n";
#ifdef _OPENMP
        #pragma omp parallel for
#endif

        // update Weibull's quantities based on the new betas
        for(unsigned int l=0; l<L; ++l)
        {
            arma::vec logMu = betas(0) + dataclass.X * betas.submat(1, l, p, l);
        }

        // save results for un-thinned posterior mean
        if(m >= burnin)
        {
            beta_post += betas;
            gamma_post += gammas;
        }
    // std::cout << "...debug19\n";
        // save results of thinned iterations
        if((m+1) % thin == 0)
        {
            beta_mcmc.row(1+nIter_thin_count) = arma::vectorise(betas).t();
            tauSq_mcmc[1+nIter_thin_count] = tauSq[0];//hyperpar.tauSq; // TODO: only keep the firs one for now

            gamma_mcmc.row(1+nIter_thin_count) = arma::vectorise(gammas).t();

            BVS_Sampler::loglikelihood(
                betas,
                kappa,
                dataclass,
                loglik
            );
    // std::cout << "...debug20\n";
            // save loglikelihoods
            loglikelihood_mcmc.row(1+nIter_thin_count) = loglik.t();

            ++nIter_thin_count;
        }

    }

}

