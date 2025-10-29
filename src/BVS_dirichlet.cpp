/* Log-likelihood for the use in Metropolis-Hastings sampler*/

#include "simple_gibbs.h"
#include "arms_gibbs.h"
#include "BVS_subfunc.h"
#include "BVS_dirichlet.h"

void BVS_dirichlet::mcmc(
    unsigned int nIter,
    unsigned int burnin,
    unsigned int thin,
    double& tau0Sq,
    arma::vec& tauSq,
    arma::mat& betas,
    arma::umat& gammas,
    const std::string& gammaProposal,
    Gamma_Sampler_Type gammaSampler,
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

    arma::mat logP_gamma = arma::zeros<arma::mat>(p, L);; // this is declared to be updated in the M-H sampler for gammas

    // std::cout << "...debug12\n";
    gamma_acc_count = 0;
    for(unsigned int l=0; l<L; ++l)
    {
        double pi = R::rbeta(hyperpar.piA, hyperpar.piB);

        for(unsigned int j=1; j<p; ++j)
        {
            gammas(j, l) = R::rbinom(1, pi);
            logP_gamma(j, l) = BVS_subfunc::logPDFBernoulli(gammas(j, l), pi);
        }
    }
    // std::cout << "...debug13\n";
    arma::vec loglik = arma::zeros<arma::vec>(N);
    loglikelihood(
        betas,
        dataclass,
        loglik
    );
    loglikelihood_mcmc.row(0) = loglik.t();
    // std::cout << "...debug14\n";
    arma::mat proportion;
    arma::mat alphas = arma::zeros<arma::mat>(N, L);
    for(unsigned int l=0; l<L; ++l)
    {
        alphas.col(l) = arma::exp( dataclass.X * betas.col(l) );
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
            tau0Sq,
            tauSq,
            dataclass
        );
        // std::cout << "...debug18\n";
#ifdef _OPENMP
        #pragma omp parallel for
#endif

        // update quantities based on the new betas
        for(unsigned int l=0; l<L; ++l)
        {
            //arma::vec logMu = dataclass.X * betas.col(l);

            // update coefficient's variances
            tauSq[l] = sampleTau(hyperpar.tauA, hyperpar.tauB, betas.submat(1,l,p-1,l));
        }
        tau0Sq = sampleTau(hyperpar.tau0A, hyperpar.tau0B, betas.row(0).t());

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

            loglikelihood(
                betas,
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


// individual loglikelihoods
void BVS_dirichlet::loglikelihood(
    const arma::mat& betas,
    const DataClass &dataclass,
    arma::vec& loglik)
{
    // dimensions
    unsigned int N = dataclass.X.n_rows;
    unsigned int L = dataclass.y.n_cols;

    arma::mat alphas = arma::zeros<arma::mat>(N, L);
    arma::vec alphas_Rowsum;
#ifdef _OPENMP
    #pragma omp parallel for
#endif

    for(unsigned int l=0; l<L; ++l)
    {
        alphas.col(l) = arma::exp( dataclass.X * betas.col(l) );
    }
    alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
    alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
    alphas_Rowsum = arma::sum(alphas, 1);

    loglik = arma::lgamma(alphas_Rowsum) - arma::sum(arma::lgamma(alphas), 1) +
             arma::sum( (alphas - 1.0) % arma::log(dataclass.y), 1 );

}


void BVS_dirichlet::sampleGamma(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    arma::vec& loglik,
    const armsParmClass& armsPar,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    double tau0Sq,
    arma::vec& tauSq,

    const DataClass &dataclass)
{

    arma::umat proposedGamma = gammas; // copy the original gammas and later change the address of the copied one
    arma::mat proposedGammaPrior;
    arma::uvec updateIdx;

    double logProposalRatio = 0;

    unsigned int N = dataclass.y.n_rows;
    unsigned int p = gammas.n_rows - 1;
    unsigned int L = gammas.n_cols;

    // define static variables for global updates for the use of bandit algorithm
    // initial value 0.5 here forces shrinkage toward 0 or 1
    static arma::mat banditAlpha = arma::mat(p, L, arma::fill::value(0.5));
    static arma::mat banditBeta = arma::mat(p, L, arma::fill::value(0.5));

    // decide on one component
    unsigned int componentUpdateIdx = 0;
    if (L > 1)
    {
        componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    }
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

    for(auto i: updateIdx)
    {
        double pi = R::rbeta(hyperpar.piA + (double)(proposedGamma(1+i,componentUpdateIdx)),
                             hyperpar.piB + (double)(p) - (double)(proposedGamma(1+i,componentUpdateIdx)));
        proposedGammaPrior(1+i,componentUpdateIdx) = BVS_subfunc::logPDFBernoulli( proposedGamma(1+i,componentUpdateIdx), pi );
        logProposalGammaRatio +=  proposedGammaPrior(1+i, componentUpdateIdx) - logP_gamma(1+i, componentUpdateIdx);
    }

    arma::mat proposedBeta = betas;
    proposedBeta.elem(arma::find(proposedGamma == 0)).fill(0.); 

    // update (addresses) 'proposedBeta' and 'logPosteriorBeta_proposal' based on 'proposedGamma'

    ARMS_Gibbs::arms_gibbs_betaK_dirichlet(
        componentUpdateIdx,
        armsPar,
        hyperpar,
        proposedBeta,
        proposedGamma,
        tau0Sq, // here fixed tau0Sq
        tauSq[componentUpdateIdx], // here fixed tauSq
        dataclass
    );


    // compute logLikelihoodRatio, i.e. proposedLikelihood - loglik
    arma::vec proposedLikelihood = loglik;
    loglikelihood( betas, dataclass, loglik );
    loglikelihood( proposedBeta, dataclass, proposedLikelihood );

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
                banditAlpha(iter,componentUpdateIdx) += banditIncrement * gammas(1+iter,componentUpdateIdx);
                banditBeta(iter,componentUpdateIdx) += banditIncrement * (1-gammas(1+iter,componentUpdateIdx));
            }

        }
    }

    // return gammas;
}

void BVS_dirichlet::sampleGammaProposalRatio(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    arma::vec& loglik,
    const armsParmClass& armsPar,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    double tau0Sq,
    arma::vec& tauSq,

    const DataClass &dataclass)
{
    arma::umat proposedGamma = gammas; // copy the original gammas and later change the address of the copied one
    arma::mat proposedGammaPrior;
    arma::uvec updateIdx;

    double logProposalRatio = 0;

    unsigned int N = dataclass.y.n_rows;
    unsigned int p = gammas.n_rows - 1;
    unsigned int L = gammas.n_cols;

    // define static variables for global updates for the use of bandit algorithm
    // initial value 0.5 here forces shrinkage toward 0 or 1
    static arma::mat banditAlpha = arma::mat(p, L, arma::fill::value(0.5));
    static arma::mat banditBeta = arma::mat(p, L, arma::fill::value(0.5));

    // decide on one component
    unsigned int componentUpdateIdx = 0;
    if (L > 1)
    {
        componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    }
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

    for(auto i: updateIdx)
    {
        double pi = R::rbeta(hyperpar.piA + (double)(proposedGamma(1+i,componentUpdateIdx)),
                             hyperpar.piB + (double)(p) - (double)(proposedGamma(1+i,componentUpdateIdx)));
        proposedGammaPrior(1+i,componentUpdateIdx) = BVS_subfunc::logPDFBernoulli( proposedGamma(1+i,componentUpdateIdx), pi );
        logProposalGammaRatio +=  proposedGammaPrior(1+i, componentUpdateIdx) - logP_gamma(1+i, componentUpdateIdx);
    }

    arma::mat proposedBeta = betas;
    proposedBeta.elem(arma::find(proposedGamma == 0)).fill(0.); 

    ARMS_Gibbs::arms_gibbs_betaK_dirichlet(
        componentUpdateIdx,
        armsPar,
        hyperpar,
        proposedBeta,
        proposedGamma,
        tau0Sq, // here fixed tau0Sq
        tauSq[componentUpdateIdx], // here fixed tauSq
        dataclass
    );

    // compute logLikelihoodRatio, i.e. proposedLikelihood - loglik
    arma::vec proposedLikelihood = loglik;
    loglikelihood( betas, dataclass, loglik );
    loglikelihood( proposedBeta, dataclass, proposedLikelihood );

    double logLikelihoodRatio = arma::sum(proposedLikelihood - loglik);

    // update density of beta priors
    double logP_beta = 0.;
    double proposedBetaPrior = 0.;
    logP_beta = logPBeta(
                    betas,
                    tauSq,
                    dataclass
                );
    proposedBetaPrior = logPBeta(
                            proposedBeta,
                            tauSq,
                            dataclass
                        );
    //// the following is the same as using logPBeta() above
    // double logP_beta = loglik + logprior;
    // double proposedBetaPrior = proposedLikelihood + proposedLogprior;

    double logPriorBetaRatio = BVS_subfunc::logPDFNormal(proposedBeta.col(componentUpdateIdx), tauSq[componentUpdateIdx]) -
                               BVS_subfunc::logPDFNormal(betas.col(componentUpdateIdx), tauSq[componentUpdateIdx]);
    double logProposalBetaRatio = logP_beta - proposedBetaPrior;

    // TODO: check if 'logProposalBetaRatio == -(logLikelihoodRatio + logPriorBetaRatio)'

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
                banditAlpha(iter,componentUpdateIdx) += banditIncrement * gammas(1+iter,componentUpdateIdx);
                banditBeta(iter,componentUpdateIdx) += banditIncrement * (1-gammas(1+iter,componentUpdateIdx));
            }

        }
    }
    // return gammas;
}

double BVS_dirichlet::logPBeta(
    const arma::mat& betas,
    const arma::vec& tauSq,
    const DataClass& dataclass
)
{
    // dimensions
    unsigned int N = dataclass.X.n_rows;
    unsigned int L = dataclass.y.n_cols;

    arma::mat alphas = arma::zeros<arma::mat>(N, L);
    arma::vec alphas_Rowsum;
    double logprior = 0.;

#ifdef _OPENMP
    #pragma omp parallel for
#endif

    for(unsigned int l=0; l<L; ++l)
    {
        alphas.col(l) = arma::exp( dataclass.X * betas.col(l) );
        logprior +=  -arma::sum(betas.col(l) % betas.col(l)) / tauSq[l] / 2.;
    }
    alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
    alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
    alphas_Rowsum = arma::sum(alphas, 1);

    double loglik = arma::sum(
        arma::lgamma(alphas_Rowsum) - 
        arma::sum(arma::lgamma(alphas), 1) + 
        arma::sum( (alphas - 1.0) % arma::log(dataclass.y), 1 )
    );


    double logP = loglik + logprior;

    return logP;
}
