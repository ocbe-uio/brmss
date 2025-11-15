/* Log-likelihood for the use in Metropolis-Hastings sampler*/

#include "simple_gibbs.h"
#include "BVS_subfunc.h"
#include "BVS_gaussian.h"

void BVS_gaussian::mcmc(
    unsigned int nIter,
    unsigned int burnin,
    unsigned int thin,
    double sigmaSq,
    double& tau0Sq,
    arma::vec& tauSq,
    arma::mat& betas,
    arma::umat& gammas,
    const std::string& gammaProposal,
    Gamma_Sampler_Type gammaSampler,
    Gamma_Gibbs_Type gammaGibbs,
    const hyperparClass& hyperpar,
    const DataClass &dataclass,

    arma::vec& sigmaSq_mcmc,
    double& sigmaSq_post,
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
    arma::mat mu = dataclass.X * betas;

    arma::vec loglik = arma::zeros<arma::vec>(N);
    loglikelihood(
        betas,
        sigmaSq,
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

        // update coefficient's variance
        tau0Sq = sampleTau0(hyperpar.tau0A, hyperpar.tau0B, betas[0]);
        tauSq = sampleTau(hyperpar.tauA, hyperpar.tauB, betas.rows(1,p-1));

        // update Gaussian's shape parameter
        sigmaSq = gibbs_sigmaSq(
                      hyperpar.sigmaA,
                      hyperpar.sigmaB,
                      dataclass,
                      mu
                  );

        // update \gammas -- variable selection indicators
        switch( gammaGibbs )
        {
        case Gamma_Gibbs_Type::none :

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
                    sigmaSq,
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
                    sigmaSq,
                    tau0Sq,
                    tauSq,

                    dataclass
                );
            }
            break;

        case Gamma_Gibbs_Type::independent :
        {

            // Gibbs sampling for gammas as Kuo & Mallick (1998, Sankhyā)
            // betas have independent spike-and-slab priors

            int n_updates = std::min(10., std::ceil( (double)(p-1) ));
            Rcpp::IntegerVector entireIdx = Rcpp::seq(1, p-1);
            // random order of indexes for better mixing. Note that here 'updateIdx' is different from the one in 'sampleGamma()'
            arma::uvec updateIdx = Rcpp::as<arma::uvec>(Rcpp::sample(entireIdx, n_updates, false)); // here 'replace = false'

            double pj = hyperpar.pj; //0.5;
            for (auto j : updateIdx)
            {
                arma::mat thetaStar = betas;
                thetaStar.elem(arma::find(gammas == 0)).fill(0.);
                arma::mat thetaStarStar = thetaStar;
                thetaStar(j) = betas(j);
                thetaStarStar(j) = 0.;

                double quad = arma::as_scalar((dataclass.y - dataclass.X * thetaStar).t() *
                                              (dataclass.y - dataclass.X * thetaStar));
                double c_j = pj * std::exp(-0.5/sigmaSq * quad);
                quad = arma::as_scalar((dataclass.y - dataclass.X * thetaStarStar).t() *
                                       (dataclass.y - dataclass.X * thetaStarStar));
                double d_j = (1.-pj) * std::exp(-0.5/sigmaSq * quad);

                double pTilde = c_j/(c_j+d_j);
                gammas(j) = R::rbinom(1, pTilde);
            }
            break;
        }
        case Gamma_Gibbs_Type::gprior :

            // Gibbs sampling for gammas as George & McCulloch (1997, Statistica Sinica)
            // TODO: betas have g-prior
            // TODO: remember to change 'gibbs_beta_gaussian()' with g-prior variance matrix for betas

            throw std::runtime_error("GLM with g-prior has not yet been implemented!");
            break;

        }

        // update \betas

        // betas %= arma::conv_to<arma::mat>::from(arma::join_cols(arma::ones<arma::urowvec>(1), gammas));
        betas.elem(arma::find(gammas == 0)).fill(0.);

        // TODO: test if the following re-update of betas is necessary, since 'sampleGamma()' update both gammas & betas
        // (void)gibbs_beta_gaussian()
        logP_beta = gibbs_beta_gaussian(
                        betas,
                        gammas,
                        tau0Sq,
                        tauSq[0],
                        sigmaSq,
                        dataclass
                    );

        mu = dataclass.X * betas;

        // save results for un-thinned posterior mean
        if(m >= burnin)
        {
            sigmaSq_post += sigmaSq;
            beta_post += betas;
            gamma_post += gammas;
        }

        // save results of thinned iterations
        if((m+1) % thin == 0)
        {
            sigmaSq_mcmc[1+nIter_thin_count] = sigmaSq;
            beta_mcmc.row(1+nIter_thin_count) = arma::vectorise(betas).t();
            tauSq_mcmc[1+nIter_thin_count] = tauSq[0];//hyperpar.tauSq; // TODO: only keep the firs one for now

            gamma_mcmc.row(1+nIter_thin_count) = arma::vectorise(gammas).t();

            loglikelihood(
                betas,
                sigmaSq,
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
void BVS_gaussian::loglikelihood(
    const arma::mat& betas,
    double sigmaSq,
    const DataClass &dataclass,
    arma::vec& loglik)
{
    // unsigned int p = dataclass.X.n_cols;
    // arma::vec mu = dataclass.X * betas;
    arma::mat res = dataclass.y - dataclass.X * betas;

    for(unsigned int i=1; i<loglik.n_elem; ++i)
    {
        loglik[i] = BVS_subfunc::logPDFNormal(res(i), sigmaSq);
    }

}


void BVS_gaussian::sampleGamma(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    arma::vec& loglik,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    double sigmaSq,
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
        // the following pi is wrong for pi_j; see below
        // double pi = R::rbeta(hyperpar.piA + (double)(proposedGamma(1+i,componentUpdateIdx)),
        //                      hyperpar.piB + (double)(p) - (double)(proposedGamma(1+i,componentUpdateIdx)));
        double pi = R::rbeta(hyperpar.piA + (double)(gammas(1+i,componentUpdateIdx)),
                             hyperpar.piB + 1 - (double)(gammas(1+i,componentUpdateIdx)));
        proposedGammaPrior(1+i,componentUpdateIdx) = BVS_subfunc::logPDFBernoulli( proposedGamma(1+i,componentUpdateIdx), pi );
        logProposalGammaRatio +=  proposedGammaPrior(1+i, componentUpdateIdx) - logP_gamma(1+i, componentUpdateIdx);
    }

    arma::mat proposedBeta = betas;
    proposedBeta.elem(arma::find(proposedGamma == 0)).fill(0.);

    (void)gibbs_beta_gaussian(
        proposedBeta,
        proposedGamma,
        tau0Sq,
        tauSq[0],
        sigmaSq,
        dataclass
    );

    // compute logLikelihoodRatio, i.e. proposedLikelihood - loglik
    arma::vec proposedLikelihood = loglik;
    loglikelihood( betas, sigmaSq, dataclass, loglik );
    loglikelihood( proposedBeta, sigmaSq, dataclass, proposedLikelihood );

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

void BVS_gaussian::sampleGammaProposalRatio(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    double& logP_beta,
    arma::vec& loglik,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    double sigmaSq,
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
    proposedBetaPrior = gibbs_beta_gaussian(
                            proposedBeta,
                            proposedGamma,
                            tau0Sq,
                            tauSq[0],
                            sigmaSq,
                            dataclass
                        );
    /*
    logPosteriorBeta = logP_beta(
                           betas,
                           tau0Sq,
                           tauSq[0],
                           sigmaSq,
                           dataclass
                       );
    */

    double logPriorBetaRatio = BVS_subfunc::logPDFNormal(proposedBeta, tauSq[0]) - BVS_subfunc::logPDFNormal(betas, tauSq[0]);
    double logProposalBetaRatio = proposedBetaPrior - logP_beta;


    // compute logLikelihoodRatio, i.e. proposedLikelihood - loglik
    arma::vec proposedLikelihood = loglik;
    loglikelihood( betas, sigmaSq, dataclass, loglik );
    loglikelihood( proposedBeta, sigmaSq, dataclass, proposedLikelihood );

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

double BVS_gaussian::gibbs_sigmaSq(
    double a,
    double b,
    const DataClass& dataclass,
    const arma::vec& mu
)
{
    a += 0.5 * (double)(dataclass.y.n_elem);

    b += 0.5 * arma::as_scalar( (dataclass.y - mu).t() * (dataclass.y - mu) );

    return ( 1. / R::rgamma(a, 1. / b) );
}


double BVS_gaussian::gibbs_beta_gaussian(
    arma::mat& betas,
    const arma::umat& gammas,
    double tau0Sq,
    double tauSq,
    double sigmaSq,
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

    arma::mat invW = X_mask.t() * X_mask / sigmaSq + arma::diagmat(diag_elements);
    arma::mat W;
    if( !arma::inv_sympd( W,  invW ) )
    {
        arma::inv(W, invW, arma::inv_opts::allow_approx);
    }

    arma::vec mu = W * X_mask.t() * dataclass.y / sigmaSq;
    arma::vec beta_mask = BVS_subfunc::randMvNormal( mu, W );
    betas(VS_idx) = beta_mask;

    double logP = BVS_subfunc::logPDFNormal( beta_mask, mu, W );

    return logP;
}

/*
double BVS_gaussian::logP_beta(
    const arma::mat& betas,
    double tau0Sq,
    double tauSq,
    double sigmaSq,
    const DataClass& dataclass)
{
    unsigned int N = dataclass.X.n_rows;
    unsigned int p = dataclass.X.n_cols;

    arma::uvec VS_idx = arma::find(betas.rows(1,p));
    arma::mat X_mask = arma::join_rows(arma::ones<arma::vec>(N), dataclass.X.cols(VS_idx));

    arma::vec diag_elements = arma::vec(VS_idx.n_elem + 1, arma::fill::value(1./tauSq));
    diag_elements[0] = 1./tau0Sq;

    arma::mat invW = X_mask.t() * X_mask / sigmaSq + arma::diagmat(diag_elements);

    arma::mat W;
    if( !arma::inv_sympd( W,  invW ) )
    {
        arma::inv(W, invW, arma::inv_opts::allow_approx);
    }

    arma::vec mu = W * X_mask.t() * dataclass.y / sigmaSq;
    // arma::vec beta_mask = randMvNormal( mu, W );

    double logP = 0.;
    // logP = logPDFNormal( betas.rows(VS_idx), mu, W );
    logP = logPDFNormal( betas.elem(arma::find(betas)), mu, W );

    return logP;
}
*/
