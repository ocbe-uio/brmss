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

    arma::mat logP_gamma = arma::zeros<arma::mat>(p, L);; // this is declared to be updated in the M-H sampler for gammas

    gamma_acc_count = 0;
    double pi = R::rbeta(hyperpar.piA, hyperpar.piB);

    for(unsigned int j=0; j<p; ++j)
    {
        gammas(j) = R::rbinom(1, pi);
        logP_gamma(j) = BVS_subfunc::logPDFBernoulli(gammas(j), pi);
    }
    double logP_beta = 0.;

    // mean parameter
    arma::mat mu = betas(0) + dataclass.X * betas.rows(1, p);

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

        // update coefficient's variance
        tau0Sq = sampleTau0(hyperpar.tau0A, hyperpar.tau0B, betas[0]);
        tauSq = sampleTau(hyperpar.tauA, hyperpar.tauB, betas.rows(1,p));

        // update Gaussian's shape parameter
        sigmaSq = gibbs_sigmaSq(
                      hyperpar.sigmaA,
                      hyperpar.sigmaB,
                      dataclass,
                      mu
                  );

        // update \gammas -- variable selection indicators
        if (gammaSampler != Gamma_Sampler_Type::gibbs) 
        {
            if (gammaProposal == "simple")
            {
                sampleGamma(
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
        } 
        else
        {
            // Gibbs sampling for gammas as Kuo & Mallick (1998, Sankhyā)

            int n_updates = std::min(10., std::ceil( (double)p ));
            Rcpp::IntegerVector entireIdx = Rcpp::seq( 0, p - 1);
            // random order of indexes for better mixing
            arma::uvec updateIdx = Rcpp::as<arma::uvec>(Rcpp::sample(entireIdx, n_updates, false)); // here 'replace = false'
            
            double pi = 0.5;
            for (auto j : updateIdx)
            {
                arma::mat thetaStar = betas;
                thetaStar.elem(1 + arma::find(gammas == 0)).fill(0.);
                arma::mat thetaStarStar = thetaStar;
                thetaStar(j) = betas(j);
                thetaStarStar(j) = 0.;

                double quad = arma::as_scalar((dataclass.y - thetaStar(0) - dataclass.X * thetaStar.rows(1,p)).t() * 
                    (dataclass.y - thetaStar(0) - dataclass.X * thetaStar.rows(1,p)));
                double c_j = pi * std::exp(-0.5/sigmaSq * quad);
                quad = arma::as_scalar((dataclass.y - thetaStarStar(0) - dataclass.X * thetaStarStar.rows(1,p)).t() * 
                    (dataclass.y - thetaStarStar(0) - dataclass.X * thetaStarStar.rows(1,p)));
                double d_j = (1.-pi) * std::exp(-0.5/sigmaSq * quad);

                double pTilde = c_j/(c_j+d_j);
                gammas(j) = R::rbinom(1, pTilde);
            }
        }

        // update \betas

        // betas %= arma::conv_to<arma::mat>::from(arma::join_cols(arma::ones<arma::urowvec>(1), gammas));
        betas.elem(1 + arma::find(gammas == 0)).fill(0.); // +1 due to intercept in betas

        // (void)gibbs_beta_gaussian(
        logP_beta = gibbs_beta_gaussian(
                       betas,
                       gammas,
                       tau0Sq,
                       tauSq[0],
                       sigmaSq,
                       dataclass
                   );

        mu = betas(0) + dataclass.X * betas.rows(1, p);

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
    unsigned int p = dataclass.X.n_cols;

    arma::vec mu = betas(0) + dataclass.X * betas.rows(1, p);
    arma::vec res= dataclass.y - mu;

    for(unsigned int i=1; i<loglik.n_elem; ++i)
    {
        loglik[i] = BVS_subfunc::logPDFNormal(res[i], sigmaSq);
    }

}


void BVS_gaussian::sampleGamma(
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

    case Gamma_Sampler_Type::gibbs:
        // Impossible in this case!
        ::Rf_error("Something going wrong in file 'src/BVS_gaussian.cpp'!");
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
    proposedBeta.elem(1 + arma::find(proposedGamma == 0)).fill(0.); // +1 due to intercept in betas
    // (void)gibbs_beta_gaussian()
    double proposedBetaPrior = gibbs_beta_gaussian(
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
                banditAlpha(iter,componentUpdateIdx) += banditIncrement * gammas(iter,componentUpdateIdx);
                banditBeta(iter,componentUpdateIdx) += banditIncrement * (1-gammas(iter,componentUpdateIdx));
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

    case Gamma_Sampler_Type::gibbs:
        // Impossible in this case!
        ::Rf_error("Something going wrong in file 'src/BVS_gaussian.cpp'!");
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
    // double logPosteriorBeta = 0.;
    double proposedBetaPrior = 0.;

    proposedBeta.elem(1 + arma::find(proposedGamma == 0)).fill(0.); // +1 due to intercept in betas
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
    double logProposalBetaRatio = logP_beta - proposedBetaPrior;


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
                banditAlpha(iter,componentUpdateIdx) += banditIncrement * gammas(iter,componentUpdateIdx);
                banditBeta(iter,componentUpdateIdx) += banditIncrement * (1-gammas(iter,componentUpdateIdx));
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

arma::vec BVS_gaussian::randMvNormal(
    const arma::vec &m,
    const arma::mat &Sigma)
{
    unsigned int d = m.n_elem;
    //check
    if(Sigma.n_rows != d || Sigma.n_cols != d )
    {
        ::Rf_error("Dimension not matching in the multivariate normal sampler");
    }

    arma::mat A;
    arma::vec eigval;
    arma::mat eigvec;
    arma::rowvec res;

    if( arma::chol(A,Sigma) )
    {
        res = randVecNormal(d).t() * A ;
    }
    else
    {
        if( eig_sym(eigval, eigvec, Sigma) )
        {
            res = (eigvec * arma::diagmat(arma::sqrt(eigval)) * randVecNormal(d)).t();
        }
        else
        {
            ::Rf_error("randMvNorm failing because of singular Sigma matrix");
        }
    }

    return res.t() + m;
}

// n-sample normal, parameters mean and variance
arma::vec BVS_gaussian::randVecNormal(
    const unsigned int n)
{
    // arma::vec res(n);
    // for(unsigned int i=0; i<n; ++i)
    // {
    //     res(i) = R::rnorm( 0., 1. );
    // }

    arma::vec res = Rcpp::rnorm(n);
    return res;
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
    unsigned int N = dataclass.X.n_rows;

    arma::uvec VS_idx = arma::find(gammas);
    arma::mat X_mask = arma::join_rows(arma::ones<arma::vec>(N), dataclass.X.cols(VS_idx));
    
    arma::uvec interceptIdx = {0};
    VS_idx.insert_rows(0, interceptIdx);
    VS_idx += 1;
    VS_idx[0] = 0;

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
    arma::vec beta_mask = randMvNormal( mu, W );
    betas(VS_idx) = beta_mask;

    double logP = 0.;
    logP = logPDFNormal( beta_mask, mu, W );

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

double BVS_gaussian::logPDFNormal(
    const arma::vec& x,
    const arma::vec& m,
    const arma::mat& Sigma)
{
    unsigned int k = Sigma.n_cols;

    double sign, tmp;
    arma::log_det(tmp, sign, Sigma ); //sign is not importantas det SHOULD be > 0 as for positive definiteness!

    return -0.5*(double)k*log(2.*M_PI) -0.5*tmp -0.5* arma::as_scalar( (x-m).t() * arma::inv_sympd(Sigma) * (x-m) );

}