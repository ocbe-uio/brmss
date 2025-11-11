/* Log-likelihood for the use in Metropolis-Hastings sampler*/

#include "simple_gibbs.h"
#include "BVS_subfunc.h"
#include "BVS_probit.h"

void BVS_probit::mcmc(
    unsigned int nIter,
    unsigned int burnin,
    unsigned int thin,
    double& tau0Sq_,
    arma::vec& tauSq_,
    arma::mat& betas,
    arma::umat& gammas,
    const std::string& gammaProposal,
    Gamma_Sampler_Type gammaSampler,
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
    // dimensions
    unsigned int N = dataclass.X.n_rows;
    unsigned int p = dataclass.X.n_cols;
    unsigned int L = dataclass.y.n_cols;

    double tau0Sq = tau0Sq_;
    double tauSq = tauSq_[0];
    arma::mat logP_gamma = arma::zeros<arma::mat>(p, L); // this is declared to be updated in the M-H sampler for gammas

    gamma_acc_count = 0;
    double pi = R::rbeta(hyperpar.piA, hyperpar.piB);

    for(unsigned int j=1; j<p; ++j)
    {
        gammas(j) = R::rbinom(1, pi);
        logP_gamma(j) = BVS_subfunc::logPDFBernoulli(gammas(j), pi);
    }
    double logP_beta = 0.;

    // mean parameter
    // arma::vec mu = dataclass.X * betas;
    arma::vec z = zbinprobit(betas, dataclass);

    arma::vec loglik = arma::zeros<arma::vec>(N);
    loglikelihood(
        betas,
        dataclass,
        loglik
    );
    loglikelihood_mcmc.row(0) = loglik.t();


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
            Rcpp::Rcout << "\r[" <<    //'\r' aka carriage return should move printer's cursor back at the beginning of the current line
                        std::string(cTotalLength * (m + 1.) / nIter, '#') <<        // printing filled part
                        std::string(cTotalLength * (1. - (m + 1.) / nIter), '-') << // printing empty part
                        "] " << (int)((m + 1.) / nIter * 100.0) << "%\r";             // printing percentage
// std::cout << "...debug1\n";
        // update coefficient's variance
        tau0Sq = sampleTau0(hyperpar.tau0A, hyperpar.tau0B, betas[0]);
        tauSq = sampleTau(hyperpar.tauA, hyperpar.tauB, betas.rows(1,p-1));
// std::cout << "...debug2\n";
        // update latent responses
// std::cout << "...debug3\n";
        // update \gammas -- variable selection indicators
        if (gammaProposal == "simple")
        {
            sampleGamma(
                gammas,
                gammaSampler,
                logP_gamma,
                gamma_acc_count,
                loglik,
                hyperpar,
                betas,
                z,
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
                logP_beta,
                loglik,
                hyperpar,
                betas,
                z,
                tau0Sq,
                tauSq,

                dataclass
            );
        }
// std::cout << "...debug4\n";
        // update \betas

        // betas.elem(arma::find(gammas == 0)).fill(0.);
        z = zbinprobit(betas, dataclass);

        // TODO: test if the following re-update of betas is necessary, since 'sampleGamma()' update both gammas & betas
        // (void)gibbs_beta_probit()
        logP_beta = gibbs_beta_probit(
                        betas,
                        gammas,
                        tau0Sq,
                        tauSq,
                        z,
                        dataclass
                    );
        z = zbinprobit(betas, dataclass);
// std::cout << "...debug5\n";
        // save results for un-thinned posterior mean
        if(m >= burnin)
        {
            beta_post += betas;
            gamma_post += gammas;
        }

        // save results of thinned iterations
        if((m+1) % thin == 0)
        {
            beta_mcmc.row(1+nIter_thin_count) = arma::vectorise(betas).t();
            tauSq_mcmc[1+nIter_thin_count] = tauSq;//hyperpar.tauSq; // TODO: only keep the firs one for now

            gamma_mcmc.row(1+nIter_thin_count) = arma::vectorise(gammas).t();

            loglikelihood(
                betas,
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
void BVS_probit::loglikelihood(
    const arma::mat& betas,
    const DataClass &dataclass,
    arma::vec& loglik)
{
    
    arma::vec normcdf_Z = arma::normcdf(dataclass.X * betas);
    normcdf_Z.elem(arma::find(normcdf_Z > 1.0-lowerbound0)).fill(1.0-lowerbound0);
    normcdf_Z.elem(arma::find(normcdf_Z < lowerbound0)).fill(lowerbound0);

    loglik = dataclass.y % arma::log(normcdf_Z) + (1.0-dataclass.y) % arma::log(1.0-normcdf_Z) ;
    

    /*
    loglik.zeros();
    arma::mat res = dataclass.X * betas;

    for(unsigned int i=1; i<loglik.n_elem; ++i)
    {
        double normPDF = BVS_subfunc::logPDFNormal(res[i], 1.0);
        if((dataclass.y[i] == 0. && z[i] <= 0.) || (dataclass.y[i] == 1. && z[i] > 0.))
            loglik[i] =  normPDF;
    }
    */

}


void BVS_probit::sampleGamma(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    arma::vec& loglik,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    const arma::vec& z,
    double tau0Sq,
    double tauSq,

    const DataClass &dataclass)
{
// std::cout << "...debug31\n";
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
    // if (L > 1)
    // {
    //     componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    // }
    arma::uvec singleIdx_k = { componentUpdateIdx };
// std::cout << "...debug32\n";
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
// std::cout << "...debug33\n";
    // note only one outcome is updated
    // update log probabilities

    // compute logProposalGammaRatio, i.e. proposedGammaPrior - logP_gamma
    double logProposalGammaRatio = 0.;

    proposedGammaPrior = logP_gamma; // copy the original one and later change the address of the copied one

    // TODO: check if pi0 is needed
    // double pi = pi0;

    for(auto i: updateIdx)
    {
        // the following pi is wrong for pi_j; see below
        // double pi = R::rbeta(hyperpar.piA + (double)(proposedGamma(1+i,componentUpdateIdx)),
        //                      hyperpar.piB + (double)(p) - (double)(proposedGamma(1+i,componentUpdateIdx)));
        double pi = R::rbeta(hyperpar.piA + (double)(gammas(1+i,componentUpdateIdx)),
                             hyperpar.piB + 1 - (double)(gammas(1+i,componentUpdateIdx)));
        proposedGammaPrior(1+i,componentUpdateIdx) = BVS_subfunc::logPDFBernoulli( proposedGamma(1+i,componentUpdateIdx), pi );
        logProposalGammaRatio +=  proposedGammaPrior(1+i, componentUpdateIdx) - logP_gamma(1+i, componentUpdateIdx);
    }
// std::cout << "...debug34\n";
    arma::mat proposedBeta = betas;
    proposedBeta.elem(arma::find(proposedGamma == 0)).fill(0.);

    (void)gibbs_beta_probit(
        proposedBeta,
        proposedGamma,
        tau0Sq,
        tauSq,
        z,
        dataclass
    );

    // arma::vec proposedZ = zbinprobit(proposedBeta, dataclass);
// std::cout << "...debug35\n";
    // compute logLikelihoodRatio, i.e. proposedLikelihood - loglik
    arma::vec proposedLikelihood = loglik;
    loglikelihood( betas, dataclass, loglik );
    loglikelihood( proposedBeta, dataclass, proposedLikelihood );
// std::cout << "...debug36\n";
    double logLikelihoodRatio = arma::sum(proposedLikelihood - loglik);

    // Here we need always compute the proposal and original ratios, in particular the likelihood, since betas are updated
    double logAccProb = logProposalGammaRatio +
                        logLikelihoodRatio +
                        logProposalRatio;
// std::cout << "...debug37\n";
    if( std::log(R::runif(0,1)) < logAccProb )
    {
        gammas = proposedGamma;
        logP_gamma = proposedGammaPrior;
        loglik = proposedLikelihood;
        betas = proposedBeta;

        ++gamma_acc_count;
    }
// std::cout << "...debug38\n";
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
// std::cout << "...debug39\n";

    // return gammas;
}

void BVS_probit::sampleGammaProposalRatio(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    double& logP_beta,
    arma::vec& loglik,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    const arma::vec& z,
    double tau0Sq,
    double tauSq,

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
        double pi = R::rbeta(hyperpar.piA + (double)(gammas(1+i,componentUpdateIdx)),
                             hyperpar.piB + 1 - (double)(gammas(1+i,componentUpdateIdx)));
        proposedGammaPrior(1+i,componentUpdateIdx) = BVS_subfunc::logPDFBernoulli( proposedGamma(1+i,componentUpdateIdx), pi );
        logProposalGammaRatio +=  proposedGammaPrior(1+i, componentUpdateIdx) - logP_gamma(1+i, componentUpdateIdx);
    }

    arma::mat proposedBeta = betas;

    // update (addresses) 'proposedBeta' and 'logPosteriorBeta_proposal' based on 'proposedGamma'
    // double logPosteriorBeta = 0.;
    double proposedBetaPrior = 0.;
    proposedBeta.elem(arma::find(proposedGamma == 0)).fill(0.);

    // // Note that intercept is updated here
    proposedBetaPrior = gibbs_beta_probit(
                            proposedBeta,
                            proposedGamma,
                            tau0Sq,
                            tauSq,
                            z,
                            dataclass
                        );
    // arma::vec proposedZ = zbinprobit(proposedBeta, dataclass);

    double logPriorBetaRatio = BVS_subfunc::logPDFNormal(proposedBeta, tauSq) - BVS_subfunc::logPDFNormal(betas, tauSq);
    double logProposalBetaRatio = logP_beta - proposedBetaPrior;


    // compute logLikelihoodRatio, i.e. proposedLikelihood - loglik
    arma::vec proposedLikelihood = loglik;
    loglikelihood( betas, dataclass, loglik );
    loglikelihood( proposedBeta, dataclass, proposedLikelihood );

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
        logP_beta = proposedBetaPrior;
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


double BVS_probit::gibbs_beta_probit(
    arma::mat& betas,
    const arma::umat& gammas,
    double tau0Sq,
    double tauSq,
    const arma::vec& z,
    const DataClass& dataclass
)
{
    arma::uvec VS_idx = arma::find(gammas);
    arma::mat X_mask = dataclass.X.cols(VS_idx);

    // arma::vec diag_elements = arma::join_rows({tau0Sq}, 1./tauSq * arma::eye<arma::mat>(VS_idx.n_elem,VS_idx.n_elem));
    // arma::vec diag_elements = { tau0Sq };
    // diag_elements.insert_rows(1, arma::vec(VS_idx.n_elem, arma::fill::value(1./tauSq)));
    arma::vec diag_elements = arma::vec(VS_idx.n_elem, arma::fill::value(1./tauSq));
    diag_elements[0] = 1./tau0Sq;

    arma::mat invW = X_mask.t() * X_mask + arma::diagmat(diag_elements);
    arma::mat W;
    if( !arma::inv_sympd( W,  invW ) )
    {
        arma::inv(W, invW, arma::inv_opts::allow_approx);
    }

    arma::vec mu = W * X_mask.t() * z;
    arma::vec beta_mask = BVS_subfunc::randMvNormal( mu, W );
    betas(VS_idx) = beta_mask;

    double logP = BVS_subfunc::logPDFNormal( beta_mask, mu, W );

    return logP;
}

arma::vec BVS_probit::zbinprobit(
    const arma::mat& betas,
    const DataClass& dataclass
)
{
    arma::vec y = dataclass.y.col(0);
    arma::vec m = dataclass.X * betas;
    unsigned int n = m.n_elem;

    arma::vec u = Rcpp::runif( n ); // don't use arma::randu() due to reproduciability
    arma::vec cd = arma::normcdf( -m );

    cd.elem(arma::find(cd > 1.0-lowerbound0)).fill(1.0-lowerbound0);
    cd.elem(arma::find(cd < lowerbound0)).fill(lowerbound0);
    arma::vec pu = (u % cd) % (1.0 - 2.0 * y) + (u + cd) % y;
    // arma::vec cpui = Rcpp::qnorm(Rcpp::as<Rcpp::NumericVector>(pu));
    arma::vec cpui = Rcpp::qnorm( Rcpp::NumericVector(Rcpp::wrap(pu)) );
    arma::vec z = m + cpui;

    return z;
}
