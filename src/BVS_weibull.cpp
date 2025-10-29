/* Log-likelihood for the use in Metropolis-Hastings sampler*/

#include "simple_gibbs.h"
#include "arms_gibbs.h"
#include "BVS_subfunc.h"
#include "BVS_weibull.h"

void BVS_weibull::mcmc(
    unsigned int nIter,
    unsigned int burnin,
    unsigned int thin,
    double kappa,
    double& tau0Sq,
    arma::vec& tauSq,
    arma::mat& betas,
    arma::umat& gammas,
    const std::string& gammaProposal,
    Gamma_Sampler_Type gammaSampler,
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
    double pi = R::rbeta(hyperpar.piA, hyperpar.piB);

    for(unsigned int j=0; j<p; ++j)
    {
        gammas(j) = R::rbinom(1, pi);
        logP_gamma(j) = BVS_subfunc::logPDFBernoulli(gammas(j), pi);
    }

    arma::vec loglik = arma::zeros<arma::vec>(N);
    loglikelihood(
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

    loglikelihood(
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


        tau0Sq = sampleTau0(hyperpar.tau0A, hyperpar.tau0B, betas[0]); // TODO: it seems not better if fixing tau0Sq=10
        tauSq[0] = sampleTau(hyperpar.tauA, hyperpar.tauB, betas.rows(1,p));

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
        if (gammaProposal == "simple")
        {
            sampleGamma(
                gammas,
                gammaSampler,
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
            sampleGammaProposalRatio(
                gammas,
                gammaSampler,
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
            tau0Sq,
            tauSq[0],

            kappa,
            dataclass
        );

        logMu = betas(0) + dataclass.X * betas.rows(1, p);
        // update \betas' variance tauSq
        // hyperpar.tauSq = sampleTau(hyperpar.tauA, hyperpar.tauB, betas);
        // tauSq_mcmc[1+m] = hyperpar.tauSq;

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
            tauSq_mcmc[1+nIter_thin_count] = tauSq[0];//hyperpar.tauSq; // TODO: only keep the firs one for now

            gamma_mcmc.row(1+nIter_thin_count) = arma::vectorise(gammas).t();

            loglikelihood(
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

// individual loglikelihoods
void BVS_weibull::loglikelihood(
    const arma::mat& betas,
    double kappa,
    const DataClass &dataclass,
    arma::vec& loglik)
{
    // dimensions
    unsigned int p = betas.n_rows - 1;

    // arma::vec mu = arma::exp( betas(0) + dataclass.X * betas.submat(1, 0, p, 0) );
    arma::vec logMu = betas(0) + dataclass.X * betas.submat(1, 0, p, 0);
    // logMu.elem(arma::find(logMu > upperbound)).fill(upperbound);
    arma::vec mu = arma::exp( logMu );
    arma::vec lambdas = mu / std::tgamma(1. + 1./kappa);

    arma::vec first_part = std::log(kappa) - kappa * (logMu - std::lgamma(1. + 1./kappa)) + //arma::log(lambdas) +
                           (kappa - 1) * arma::log(dataclass.y);// + weibull_logS;
    first_part.elem(arma::find(dataclass.event == 0)).fill(0.);

    arma::vec second_part =  - arma::pow( dataclass.y / lambdas, kappa);

    loglik = first_part + second_part;

}


void BVS_weibull::sampleGamma(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    arma::vec& loglik,
    const armsParmClass& armsPar,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    double kappa,
    double tau0Sq,
    arma::vec& tauSq,

    const DataClass &dataclass)
{

    arma::umat proposedGamma = gammas; // copy the original gammas and later change the address of the copied one
    arma::mat proposedGammaPrior;
    arma::uvec updateIdx;

    double logProposalRatio = 0;

    unsigned int N = dataclass.y.n_rows;
    unsigned int p = gammas.n_rows;
    unsigned int L = gammas.n_cols;

    // define static variables for global updates for the use of bandit algorithm
    // initial value 0.5 here forces shrinkage toward 0 or 1
    static arma::mat banditAlpha = arma::mat(p, L, arma::fill::value(0.5));
    static arma::mat banditBeta = arma::mat(p, L, arma::fill::value(0.5));

    // decide on one component
    unsigned int componentUpdateIdx = 0;
    // if (L > 1)
    // {
    //     componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    // }
    arma::uvec singleIdx_k = { componentUpdateIdx };

    // Update the proposed Gamma with 'updateIdx' renewed via its address
    switch( gamma_sampler )
    {
    case Gamma_Sampler_Type::bandit:
        logProposalRatio += BVS_subfunc::gammaBanditProposal( p, proposedGamma, gammas, updateIdx, componentUpdateIdx, banditAlpha );
        break;

    case Gamma_Sampler_Type::mc3:
        logProposalRatio += BVS_subfunc::gammaMC3Proposal( p, proposedGamma, gammas, updateIdx, componentUpdateIdx );
        break;
    }

    // note only one outcome is updated
    // update log probabilities

    // compute logProposalGammaRatio, i.e. proposedGammaPrior - logP_gamma
    double logProposalGammaRatio = 0.;

    proposedGammaPrior = logP_gamma; // copy the original one and later change the address of the copied one

    // TODO: check if pi0 is needed
    // double pi = pi0;
    for(auto i: updateIdx)
    {
        double pi = R::rbeta(hyperpar.piA + (double)(proposedGamma(i,componentUpdateIdx)),
                             hyperpar.piB + (double)(p) - (double)(proposedGamma(i,componentUpdateIdx)));
        proposedGammaPrior(i,componentUpdateIdx) = BVS_subfunc::logPDFBernoulli( proposedGamma(i,componentUpdateIdx), pi );
        logProposalGammaRatio +=  proposedGammaPrior(i, componentUpdateIdx) - logP_gamma(i, componentUpdateIdx);
    }

    arma::mat proposedBeta = betas;

    // update (addresses) 'proposedBeta' and 'logPosteriorBeta_proposal' based on 'proposedGamma'

    ARMS_Gibbs::arms_gibbs_beta_weibull(
        armsPar,
        hyperpar,
        proposedBeta,
        proposedGamma,
        tau0Sq,
        tauSq[0],
        kappa,
        dataclass
    );

    // compute logLikelihoodRatio, i.e. proposedLikelihood - loglik
    arma::vec proposedLikelihood = loglik;
    loglikelihood( betas, kappa, dataclass, loglik );
    loglikelihood( proposedBeta, kappa, dataclass, proposedLikelihood );

    double logLikelihoodRatio = arma::sum(proposedLikelihood - loglik);

    // Here we need always compute the proposal and original ratios, in particular the likelihood, since betas are updated
    double logAccProb = logProposalGammaRatio +
                        logLikelihoodRatio +
                        logProposalRatio;

    if( std::log(R::runif(0,1)) < logAccProb )
    {
        gammas = proposedGamma;
        logP_gamma = proposedGammaPrior;
        loglik = proposedLikelihood;
        betas = proposedBeta;

        ++gamma_acc_count;
    }

    // after A/R, update bandit Related variables
    if( gamma_sampler == Gamma_Sampler_Type::bandit )
    {
        // banditLimit to control the beta prior with relatively large variance
        double banditLimit = (double)(N);
        double banditIncrement = 1.;

        for(auto iter: updateIdx)
        {
            // FINITE UPDATE
            if( banditAlpha(iter,componentUpdateIdx) + banditBeta(iter,componentUpdateIdx) <= banditLimit )
            {
                banditAlpha(iter,componentUpdateIdx) += banditIncrement * gammas(iter,componentUpdateIdx);
                banditBeta(iter,componentUpdateIdx) += banditIncrement * (1-gammas(iter,componentUpdateIdx));
            }

        }
    }

    // return gammas;
}

void BVS_weibull::sampleGammaProposalRatio(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    arma::vec& loglik,
    const armsParmClass& armsPar,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    double kappa,
    double tau0Sq,
    arma::vec& tauSq,

    const DataClass &dataclass)
{
    arma::umat proposedGamma = gammas; // copy the original gammas and later change the address of the copied one
    arma::mat proposedGammaPrior;
    arma::uvec updateIdx;

    double logProposalRatio = 0;

    unsigned int N = dataclass.y.n_rows;
    unsigned int p = gammas.n_rows;
    unsigned int L = gammas.n_cols;

    // define static variables for global updates for the use of bandit algorithm
    // initial value 0.5 here forces shrinkage toward 0 or 1
    static arma::mat banditAlpha = arma::mat(p, L, arma::fill::value(0.5));
    static arma::mat banditBeta = arma::mat(p, L, arma::fill::value(0.5));

    // decide on one component
    unsigned int componentUpdateIdx = 0;
    // if (L > 1)
    // {
    //     componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    // }
    arma::uvec singleIdx_k = { componentUpdateIdx };

    // Update the proposed Gamma with 'updateIdx' renewed via its address
    switch( gamma_sampler )
    {
    case Gamma_Sampler_Type::bandit:
        logProposalRatio += BVS_subfunc::gammaBanditProposal( p, proposedGamma, gammas, updateIdx, componentUpdateIdx, banditAlpha );
        break;

    case Gamma_Sampler_Type::mc3:
        logProposalRatio += BVS_subfunc::gammaMC3Proposal( p, proposedGamma, gammas, updateIdx, componentUpdateIdx );
        break;
    }

    // note only one outcome is updated
    // update log probabilities

    // compute logProposalGammaRatio, i.e. proposedGammaPrior - logP_gamma
    double logProposalGammaRatio = 0.;

    proposedGammaPrior = logP_gamma; // copy the original one and later change the address of the copied one

    // TODO: check if pi0 is needed
    // double pi = pi0;
    for(auto i: updateIdx)
    {
        double pi = R::rbeta(hyperpar.piA + (double)(proposedGamma(i,componentUpdateIdx)),
                             hyperpar.piB + (double)(p) - (double)(proposedGamma(i,componentUpdateIdx)));
        proposedGammaPrior(i,componentUpdateIdx) = BVS_subfunc::logPDFBernoulli( proposedGamma(i,componentUpdateIdx), pi );
        logProposalGammaRatio +=  proposedGammaPrior(i, componentUpdateIdx) - logP_gamma(i, componentUpdateIdx);
    }

    arma::mat proposedBeta = betas;

    // update (addresses) 'proposedBeta' and 'logPosteriorBeta_proposal' based on 'proposedGamma'
    double logPosteriorBeta = 0.;
    double logPosteriorBeta_proposal = 0.;

    ARMS_Gibbs::arms_gibbs_beta_weibull(
        armsPar,
        hyperpar,
        proposedBeta,
        proposedGamma,
        tauSq[0],
        tau0Sq,
        kappa,
        dataclass
    );

    logPosteriorBeta = logP_beta(
                           betas,
                           tauSq[0],
                           kappa,
                           dataclass
                       );
    logPosteriorBeta_proposal = logP_beta(
                                    proposedBeta,
                                    tauSq[0],
                                    kappa,
                                    dataclass
                                );

    double logPriorBetaRatio = BVS_subfunc::logPDFNormal(proposedBeta, tauSq[0]) - BVS_subfunc::logPDFNormal(betas, tauSq[0]);
    double logProposalBetaRatio = logPosteriorBeta - logPosteriorBeta_proposal;


    // compute logLikelihoodRatio, i.e. proposedLikelihood - loglik
    arma::vec proposedLikelihood = loglik;
    loglikelihood( betas, kappa, dataclass, loglik );
    loglikelihood( proposedBeta, kappa, dataclass, proposedLikelihood );

    double logLikelihoodRatio = arma::sum(proposedLikelihood - loglik);

    // Here we need always compute the proposal and original ratios, in particular the likelihood, since betas are updated
    double logAccProb = logProposalGammaRatio +
                        logLikelihoodRatio +
                        logProposalRatio +
                        logPriorBetaRatio + logProposalBetaRatio;

    if( std::log(R::runif(0,1)) < logAccProb )
    {
        gammas = proposedGamma;
        logP_gamma = proposedGammaPrior;
        loglik = proposedLikelihood;
        betas = proposedBeta;

        ++gamma_acc_count;
    }

    // after A/R, update bandit Related variables
    if( gamma_sampler == Gamma_Sampler_Type::bandit )
    {
        // banditLimit to control the beta prior with relatively large variance
        double banditLimit = (double)(N);
        double banditIncrement = 1.;

        for(auto iter: updateIdx)
        {
            // FINITE UPDATE
            if( banditAlpha(iter,componentUpdateIdx) + banditBeta(iter,componentUpdateIdx) <= banditLimit )
            {
                banditAlpha(iter,componentUpdateIdx) += banditIncrement * gammas(iter,componentUpdateIdx);
                banditBeta(iter,componentUpdateIdx) += banditIncrement * (1-gammas(iter,componentUpdateIdx));
            }

        }
    }
    // return gammas;
}


double BVS_weibull::logP_beta(
    const arma::mat& betas,
    double tauSq,
    double kappa,
    const DataClass& dataclass)
{
    unsigned int p = dataclass.X.n_cols;

    arma::vec logMu = betas(0) + dataclass.X * betas.submat(1, 0, p, 0);

    arma::vec lambdas = arma::exp(logMu) / std::tgamma(1. + 1./kappa);

    double logprior = - arma::accu(betas % betas) / tauSq / 2.;

    arma::vec logpost_first = std::log(kappa) -
                              kappa * (logMu - std::lgamma(1. + 1./kappa)) +
                              (kappa - 1) * arma::log(dataclass.y);
    double logpost_first_sum = arma::sum( logpost_first.elem(arma::find(dataclass.event == 1)) );

    arma::vec logpost_second = arma::pow( dataclass.y / lambdas, kappa);
    // logpost_second.elem(arma::find(logpost_second > upperbound2)).fill(upperbound2);
    // double logpost_second_sum =  arma::sum( - logpost_second );
    // return logpost_first_sum + logpost_second_sum + logprior;

    double h = 0.;
    if(logpost_second.has_inf())
    {
        h = - std::numeric_limits<double>::infinity();
    }
    else
    {
        h = logpost_first_sum - arma::sum( logpost_second ) + logprior;
    }

    return h;
}

