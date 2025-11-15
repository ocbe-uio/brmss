/* not yet implemented*/

#include "simple_gibbs.h"
#include "BVS_subfunc.h"
#include "BVS_dMVP.h"

#include <RcppEigen.h>

void BVS_dMVP::mcmc(
    unsigned int nIter,
    unsigned int burnin,
    unsigned int thin,
    arma::mat& betas,
    arma::umat& gammas,
    const std::string& gammaProposal,
    Gamma_Sampler_Type gammaSampler,
    const hyperparClass& hyperpar,
    const DataClass &dataclass,

    arma::vec& sigmaSq_mcmc,
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

    arma::mat logP_gamma = arma::zeros<arma::mat>(p, L); // this is declared to be updated in the M-H sampler for gammas

    // initialize relevant quantities
    arma::mat SigmaRho = arma::diagmat( arma::ones<arma::vec>(L) );
    arma::mat U = dataclass.y - dataclass.X * betas;
    arma::mat RhoU = createRhoU( U, SigmaRho );

    arma::mat Psi = arma::zeros<arma::mat>(L, L);
    double tau0Sq = 1.;
    double tauSq = 1.;
    // double psi = 1.;
    // double logP_psi = 0.;
    double logP_SigmaRho = 0.;

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
    arma::mat D = arma::diagmat(arma::ones<arma::vec>(L));
    arma::mat Z = arma::zeros<arma::mat>(N, L);
    sampleZ(Z, D, betas, SigmaRho, dataclass);

    // std::cout << "...debug13\n";
    double log_likelihood = logLikelihood( betas, D, dataclass );
    arma::vec loglik = arma::zeros<arma::vec>(N);
    loglikelihood_conditional(
        betas,
        D,
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
            Rcpp::Rcout << "\r[" <<                                           //'\r' aka carriage return should move printer's cursor back at the beginning of the current line
                        std::string(cTotalLength * (m + 1.) / nIter, '#') <<        // printing filled part
                        std::string(cTotalLength * (1. - (m + 1.) / nIter), '-') << // printing empty part
                        "] " << (int)((m + 1.) / nIter * 100.0) << "%\r";             // printing percentage


        // std::cout << "...debug15\n";
        tauSq = sampleTau(hyperpar.tauA, hyperpar.tauB, betas.submat(1,0,p-1,L-1));
        tau0Sq = sampleTau(hyperpar.tau0A, hyperpar.tau0B, betas.row(0).t());

        // update reparametrized covariance matrix
        gibbs_SigmaRho(
            SigmaRho,
            // psi,
            U,
            RhoU,
            hyperpar.nu,
            logP_SigmaRho,
            Z,
            dataclass
        );
        // /*testing*/SigmaRho = arma::diagmat( arma::ones<arma::vec>(L) );

        // random-walk MH update for psi
        // samplePsi(psi, hyperpar.psiA, hyperpar.psiB, hyperpar.nu, logP_psi, logP_SigmaRho, SigmaRho);

        // std::cout << "...debug16\n";
        // update \gammas -- variable selection indicators
        if (gammaProposal == "simple")
        {
            sampleGamma(
                gammas,
                gammaSampler,
                logP_gamma,
                gamma_acc_count,
                log_likelihood,
                hyperpar,
                betas,
                tau0Sq,
                tauSq,
                SigmaRho,
                U,
                RhoU,
                Z,
                D,
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
                log_likelihood,
                hyperpar,
                betas,
                tau0Sq,
                tauSq,
                SigmaRho,
                U,
                RhoU,
                Z,
                D,
                dataclass
            );
        }

        // std::cout << "...debug18\n";
        // betas.elem(arma::find(gammas == 0)).fill(0.);

        // update post-hoc betas
        gibbs_betas(
            betas,
            gammas,
            SigmaRho,
            U,
            RhoU,
            tau0Sq,
            tauSq,
            Z,
            dataclass
        );

        // update sort of covariance matrix Psi
        updatePsi( SigmaRho, Psi );

        // update latent response variables
        sampleZ(Z, D, betas, Psi, dataclass);

        log_likelihood = logLikelihood( betas, D, dataclass);


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
            sigmaSq_mcmc[1+nIter_thin_count] = SigmaRho(0,0); //TODO: only keep the first one for now
            // beta_mcmc.row(1+nIter_thin_count) = arma::vectorise(betas).t();
            beta_mcmc.row(1+nIter_thin_count) = arma::vectorise(betas * arma::inv_sympd(D)).t();
            tauSq_mcmc[1+nIter_thin_count] = tauSq;//hyperpar.tauSq; // TODO: only keep the firs one for now

            gamma_mcmc.row(1+nIter_thin_count) = arma::vectorise(gammas).t();

            loglikelihood_conditional(
                betas,
                D,
                dataclass,
                loglik
            );
            loglikelihood_mcmc.row(1+nIter_thin_count) = loglik.t();

            ++nIter_thin_count;
        }

    }

}


void BVS_dMVP::sampleGamma(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    double& log_likelihood,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    const double tau0Sq,
    const double tauSq,
    const arma::mat& SigmaRho,
    arma::mat& U,
    arma::mat& RhoU,

    const arma::mat& Z,
    const arma::mat& D,
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
    componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );

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
        //// response-specific Bernoulli probability
        double pi = R::rbeta(hyperpar.piA + (double)(arma::sum(gammas.row(1+i))),
                             hyperpar.piB + (double)L - (double)(arma::sum(gammas.row(1+i))));
        proposedGammaPrior(1+i,componentUpdateIdx) = BVS_subfunc::logPDFBernoulli( proposedGamma(1+i,componentUpdateIdx), pi );
        logProposalGammaRatio +=  proposedGammaPrior(1+i, componentUpdateIdx) - logP_gamma(1+i, componentUpdateIdx);
    }

    arma::mat proposedBeta = betas;
    // proposedBeta.elem(arma::find(proposedGamma == 0)).fill(0.);
    arma::mat proposedU = U;
    arma::mat proposedRhoU = RhoU;

    // update (addresses) 'proposedBeta' and 'logPosteriorBeta_proposal' based on 'proposedGamma'

    gibbs_betaK(
        componentUpdateIdx,
        proposedBeta,
        proposedGamma,
        SigmaRho,
        proposedU,
        proposedRhoU,
        tau0Sq,
        tauSq,
        Z,
        dataclass
    );

    // compute logLikelihoodRatio, i.e. proposedLikelihood - log_likelihood
    double proposedLikelihood = logLikelihood( proposedBeta, D, dataclass );

    double logLikelihoodRatio = proposedLikelihood - log_likelihood;

    // Here we need always compute the proposal and original ratios, in particular the likelihood, since betas are updated
    double logAccProb = logProposalGammaRatio +
                        logLikelihoodRatio +
                        logProposalRatio;

    if( std::log(R::runif(0,1)) < logAccProb )
    {
        gammas = proposedGamma;
        logP_gamma = proposedGammaPrior;
        log_likelihood = proposedLikelihood;
        betas = proposedBeta;
        U = proposedU;
        RhoU = proposedRhoU;

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

void BVS_dMVP::sampleGammaProposalRatio(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    double& log_likelihood,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    const double tau0Sq,
    const double tauSq,
    const arma::mat& SigmaRho,
    arma::mat& U,
    arma::mat& RhoU,

    const arma::mat& Z,
    const arma::mat& D,
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
    componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );

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

    // compute logProposalGammaRatio, i.e. proposedGammaPrior - logP_gamma
    double logProposalGammaRatio = 0.;

    proposedGammaPrior = logP_gamma; // copy the original one and later change the address of the copied one

    for(auto i: updateIdx)
    {
        double pi = R::rbeta(hyperpar.piA + (double)(arma::sum(gammas.row(1+i))),
                             hyperpar.piB + (double)L - (double)(arma::sum(gammas.row(1+i))));
        proposedGammaPrior(1+i,componentUpdateIdx) = BVS_subfunc::logPDFBernoulli( proposedGamma(1+i,componentUpdateIdx), pi );
        logProposalGammaRatio +=  proposedGammaPrior(1+i, componentUpdateIdx) - logP_gamma(1+i, componentUpdateIdx);
    }

    arma::mat proposedBeta = betas;
    // proposedBeta.elem(arma::find(proposedGamma == 0)).fill(0.);
    arma::mat proposedU = U;
    arma::mat proposedRhoU = RhoU;

    // update (addresses) 'proposedBeta' and 'logPosteriorBeta_proposal' based on 'proposedGamma'

    logProposalRatio -= gibbs_betaK(
                            componentUpdateIdx,
                            proposedBeta,
                            proposedGamma,
                            SigmaRho,
                            proposedU,
                            proposedRhoU,
                            tau0Sq,
                            tauSq,
                            Z,
                            dataclass
                        );
    logProposalRatio += logP_gibbs_betaK(
                            componentUpdateIdx,
                            betas,
                            gammas,
                            SigmaRho,
                            proposedU,
                            proposedRhoU,
                            tau0Sq,
                            tauSq,
                            Z,
                            dataclass
                        );

    double proposedLikelihood = logLikelihood( proposedBeta, D, dataclass );

    double logLikelihoodRatio = proposedLikelihood - log_likelihood;


    // update density of beta priors
    double logP_beta = logPBetaMask( betas, gammas, tau0Sq, tauSq );
    double proposedBetaPrior = logPBetaMask( proposedBeta, proposedGamma, tau0Sq, tauSq );

    double logProposalBetaRatio = proposedBetaPrior - logP_beta;


    // Here we need always compute the proposal and original ratios, in particular the likelihood, since betas are updated
    double logAccProb = logProposalGammaRatio +
                        logLikelihoodRatio +
                        logProposalRatio +
                        // logPriorBetaRatio +
                        logProposalBetaRatio;

    if( std::log(R::runif(0,1)) < logAccProb )
    {
        gammas = proposedGamma;
        logP_gamma = proposedGammaPrior;
        log_likelihood = proposedLikelihood;
        betas = proposedBeta;
        U = proposedU;
        RhoU = proposedRhoU;

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


// individual loglikelihoods
void BVS_dMVP::loglikelihood_conditional(
    const arma::mat& betas,
    const arma::mat& D,
    const DataClass &dataclass,
    arma::vec& loglik)
{

    // loglik = arma::sum( dataclass.y % arma::log(arma::normcdf(Z))  +
    //                     (1.0-dataclass.y) % arma::log(1.0-arma::normcdf(Z)), 1 );

    arma::mat normcdf_Z = arma::normcdf(dataclass.X * betas * arma::inv_sympd(D));

    // fix numerical issues if log(0)
    normcdf_Z.elem(arma::find(normcdf_Z > 1.0-lowerbound0)).fill(1.0-lowerbound0);
    normcdf_Z.elem(arma::find(normcdf_Z < lowerbound0)).fill(lowerbound0);
    loglik = arma::sum( dataclass.y % arma::log(normcdf_Z) +
                        (1.0-dataclass.y) % arma::log(1.0-normcdf_Z),
                        1 );
}

double BVS_dMVP::logLikelihood(
    const arma::mat& betas,
    const arma::mat& D,
    const DataClass &dataclass)
{
    double logP = 0.;

    arma::mat normcdf_Z = arma::normcdf(dataclass.X * betas * arma::inv_sympd(D));

    // fix numerical issues if log(0)
    normcdf_Z.elem(arma::find(normcdf_Z > 1.0-lowerbound0)).fill(1.0-lowerbound0);
    normcdf_Z.elem(arma::find(normcdf_Z < lowerbound0)).fill(lowerbound0);
    logP = arma::accu( dataclass.y % arma::log(normcdf_Z) +
                       (1.0-dataclass.y) % arma::log(1.0-normcdf_Z) );

    if(std::isnan(logP))
    {
        throw std::runtime_error("...logLikelihood() std::isnan(logP) cdf(XB): (min,max)=");
        //normcdf_Z.min() << ", " << normcdf_Z.max() << "; sum(B)=" << arma::accu(betas) << "\n";
    }

    return logP;
}

void BVS_dMVP::gibbs_betas(
    arma::mat& betas,
    const arma::umat& gammas,
    const arma::mat& SigmaRho,
    arma::mat& U,
    arma::mat& RhoU,
    const double tau0Sq,
    const double tauSq,
    const arma::mat& Z,
    const DataClass &dataclass)
{
    // double logP = 0.;
    unsigned int L = Z.n_cols;

    // arma::mat U = Z - dataclass.X * betas;
    betas.fill( 0. ); // reset here, since we already computed U above

    arma::mat y_tilde = Z - RhoU ;
    y_tilde.each_row() /= (SigmaRho.diag().t()) ; // divide each col by the corresponding element of sigma

    arma::vec xtxMultiplier = arma::zeros<arma::vec>(L);

    for( unsigned int k=0; k<L-1; ++k)
    {
        for(unsigned int l=k+1 ; l<L ; ++l)
        {
            xtxMultiplier(k) += SigmaRho(l,k) * SigmaRho(l,k) /  SigmaRho(l,l);
            y_tilde.col(k) -= (  SigmaRho(l,k) /  SigmaRho(l,l) ) *
                              ( U.col(l) - RhoU.col(l) +  SigmaRho(l,k) * ( U.col(k) - Z.col(k) ) );
        }
    }

    for(unsigned int k=0; k<L; ++k)
    {
        arma::uvec VS_IN_k = arma::find(gammas.col(k)); // include intercept

        arma::vec diag_elements = arma::vec(VS_IN_k.n_elem, arma::fill::value(1./tauSq));
        diag_elements[0] = 1./tau0Sq;

        arma::mat invW = dataclass.X.cols(VS_IN_k).t() * dataclass.X.cols(VS_IN_k) *
                         ( 1./SigmaRho(k,k) + xtxMultiplier(k)  );// + arma::diagmat(diag_elements);
        invW.diag() += diag_elements;
        arma::mat W_k;
        if( !arma::inv_sympd( W_k,  invW ) )
        {
            arma::inv(W_k, invW, arma::inv_opts::allow_approx);
        }

        arma::vec mu_k = W_k * dataclass.X.cols(VS_IN_k).t() * y_tilde.col(k);
        arma::vec beta_mask = BVS_subfunc::randMvNormal( mu_k, W_k );

        // logP += BVS_subfunc::logPDFNormal( beta_mask, mu_k, W_k );

        arma::uvec singleIdx_k = {k};
        betas(VS_IN_k, singleIdx_k) = beta_mask;
    }

    U = Z - dataclass.X * betas;
    RhoU = createRhoU(U,  SigmaRho);  // no need, because it's not used before next update in gibbs_SigmaRho()

    // return logP;
}

double BVS_dMVP::logPBetaMask(
    const arma::mat& betas,
    const arma::umat& gammas,
    const double tau0Sq,
    const double tauSq)
{
    double logP = 0.;

    unsigned int L = betas.n_cols;
    arma::uvec singleIdx_k(1);


    for(unsigned int k=0; k<L; ++k)
    {
        arma::uvec VS_IN_k = arma::find(gammas.col(k)); // include intercept

        arma::vec diag_elements = arma::vec(VS_IN_k.n_elem, arma::fill::value(tauSq));
        diag_elements[0] = tau0Sq;

        singleIdx_k[0] = k;
        logP += BVS_subfunc::logPDFNormal( betas(VS_IN_k, singleIdx_k),  arma::diagmat(diag_elements));
    }

    return logP;
}

double BVS_dMVP::gibbs_betaK(
    const unsigned int k,
    arma::mat& betas,
    const arma::umat& gammas,
    const arma::mat& SigmaRho,
    arma::mat& U,
    arma::mat& RhoU,
    const double tau0Sq,
    const double tauSq,
    const arma::mat& Z,
    const DataClass &dataclass)
{
    double logP = 0.;
    // betas.col(k).fill( 0. );

    unsigned int L = betas.n_cols;

    // arma::mat U = Z - dataclass.X * betas;
    arma::vec y_tilde = Z.col(k) - RhoU.col(k);
    y_tilde /= SigmaRho(k,k) ;

    double xtxMultiplier = 0.;
    arma::uvec xi = arma::linspace<arma::uvec>(0, L-1, L);
    // unsigned int k_idx = arma::as_scalar( arma::find( xi == k , 1 ) );
    for(unsigned int l=k+1 ; l<L ; ++l)
    {
        xtxMultiplier += SigmaRho(l,k) * SigmaRho(l,k) /  SigmaRho(l,l);
        y_tilde -= (  SigmaRho(l,k) /  SigmaRho(l,l) ) *
                   ( U.col(l) - RhoU.col(l) +  SigmaRho(l,k) * ( U.col(k) - Z.col(k) ) );
    }
    /*
    {
        xtxMultiplier += pow( SigmaRho(xi(l),k),2) /  SigmaRho(xi(l),xi(l));
        y_tilde -= (  SigmaRho(xi(l),k) /  SigmaRho(xi(l),xi(l)) ) *
            ( U.col(xi(l)) - RhoU.col(xi(l)) +  SigmaRho(xi(l),k) * ( U.col(k) - Z.col(k) ) );
    }
    */


    arma::uvec VS_IN_k = arma::find(gammas.col(k)); // include intercept

    arma::vec diag_elements = arma::vec(VS_IN_k.n_elem, arma::fill::value(1./tauSq));
    diag_elements[0] = 1./tau0Sq;

    arma::mat invW = dataclass.X.cols(VS_IN_k).t() * dataclass.X.cols(VS_IN_k) *
                     ( 1./SigmaRho(k,k) + xtxMultiplier  );// + arma::diagmat(diag_elements);
    invW.diag() += diag_elements;
    arma::mat W_k;
    if( !arma::inv_sympd( W_k,  invW ) )
    {
        arma::inv(W_k, invW, arma::inv_opts::allow_approx);
    }

    arma::vec mu_k = W_k * dataclass.X.cols(VS_IN_k).t() * y_tilde;
    arma::vec beta_mask = BVS_subfunc::randMvNormal( mu_k, W_k );

    logP = BVS_subfunc::logPDFNormal( beta_mask, mu_k, W_k );

    arma::uvec singleIdx_k = {k};
    betas.col(k).fill( 0. ); // reset 0 before giving new updated nonzero entries
    betas(VS_IN_k, singleIdx_k) = beta_mask;

    U = Z - dataclass.X * betas;
    RhoU = createRhoU( U, SigmaRho );

    return logP;
}

double BVS_dMVP::logP_gibbs_betaK(
    const unsigned int k,
    const arma::mat& betas,
    const arma::umat& gammas,
    const arma::mat& SigmaRho,
    const arma::mat& U,
    const arma::mat& RhoU,
    const double tau0Sq,
    const double tauSq,
    const arma::mat& Z,
    const DataClass &dataclass)
{
    double logP = 0.;
    // betas.col(k).fill( 0. );

    unsigned int L = betas.n_cols;

    // arma::mat U = Z - dataclass.X * betas;
    arma::vec y_tilde = Z.col(k) - RhoU.col(k);
    y_tilde /= SigmaRho(k,k) ;

    double xtxMultiplier = 0.;

    for(unsigned int l=k+1 ; l<L ; ++l)
    {
        xtxMultiplier += SigmaRho(l,k) * SigmaRho(l,k) /  SigmaRho(l,l);
        y_tilde -= (  SigmaRho(l,k) /  SigmaRho(l,l) ) *
                   ( U.col(l) - RhoU.col(l) +  SigmaRho(l,k) * ( U.col(k) - Z.col(k) ) );
    }

    arma::uvec VS_IN_k = arma::find(gammas.col(k)); // include intercept

    arma::vec diag_elements = arma::vec(VS_IN_k.n_elem, arma::fill::value(1./tauSq));
    diag_elements[0] = 1./tau0Sq;

    arma::mat invW = dataclass.X.cols(VS_IN_k).t() * dataclass.X.cols(VS_IN_k) *
                     ( 1./SigmaRho(k,k) + xtxMultiplier  );// + arma::diagmat(diag_elements);
    invW.diag() += diag_elements;
    arma::mat W_k;
    if( !arma::inv_sympd( W_k,  invW ) )
    {
        arma::inv(W_k, invW, arma::inv_opts::allow_approx);
    }

    arma::vec mu_k = W_k * dataclass.X.cols(VS_IN_k).t() * y_tilde;
    arma::uvec singleIdx_k = {k};
    logP = BVS_subfunc::logPDFNormal( betas(VS_IN_k, singleIdx_k), mu_k, W_k );

    return logP;
}

void BVS_dMVP::gibbs_SigmaRho(
    arma::mat& SigmaRho,
    // const double psi,
    const arma::mat& U,
    arma::mat& RhoU,
    const double nu,
    double& logP_SigmaRho,
    const arma::mat& Z,
    const DataClass& dataclass)
{
    double logP = 0.;

    unsigned int N = Z.n_rows;
    unsigned int L = Z.n_cols;

    SigmaRho.fill(0.);

    // arma::mat U = Z - dataclass.X * betas;
    arma::mat Sigma = U.t() * U;
    Sigma.diag() += 1.0;

    double thisSigmaTT;
    arma::uvec conditioninIndexes;
    unsigned int nConditioninIndexes;

    double a,b;
    arma::mat rhoVar; // inverse matrix of the residual elements of Sigma in the component
    arma::rowvec rhoMean; // this is the partial Schur complement, needed for the sampler

    arma::uvec singleIdx_k(1); // needed for convention with arma::submat
    for( unsigned int k=0; k<L; ++k )
    {
        singleIdx_k(0) = k;

        // start computing interesting things
        thisSigmaTT = Sigma(k,k);

        nConditioninIndexes = k;
        conditioninIndexes.zeros( nConditioninIndexes );

        if( nConditioninIndexes > 0 )
        {
            conditioninIndexes = arma::regspace<arma::uvec>(0, k-1);

            if( !arma::inv_sympd( rhoVar, Sigma(conditioninIndexes,conditioninIndexes) ) )
            {
                arma::inv(rhoVar, Sigma(conditioninIndexes,conditioninIndexes), arma::inv_opts::allow_approx);
            }
            rhoMean = Sigma( singleIdx_k, conditioninIndexes ) * rhoVar ;
            thisSigmaTT -= arma::as_scalar( rhoMean * Sigma( conditioninIndexes, singleIdx_k ) );
        }

        // *** Diagonal Element

        // Compute parameters
        a = 0.5 * N + k + 1.0; // plus 1.0 due to C++ indexing k from 0
        // a = 0.5 * ( N + nu - L + 2*nConditioninIndexes + 1. ) ;
        b = 0.5 * thisSigmaTT ;
        // if(b <= 0 )
        // {
        //     throw std::runtime_error(" Negative parameter in gibbs_SigmaRho b");
        // }

        SigmaRho(k,k) = BVS_subfunc::randIGamma( a, b );

        logP += BVS_subfunc::logPDFIGamma( SigmaRho(k,k), a, b );


        // *** Off-Diagonal Element(s)
        if( nConditioninIndexes > 0 )
        {
            SigmaRho( conditioninIndexes, singleIdx_k ) =
                BVS_subfunc::randMvNormal( rhoMean.t(), SigmaRho(k,k) * rhoVar );

            SigmaRho( singleIdx_k, conditioninIndexes ) =
                SigmaRho( conditioninIndexes, singleIdx_k ).t();

            logP += BVS_subfunc::logPDFNormal( SigmaRho( conditioninIndexes, singleIdx_k ), rhoMean.t(), SigmaRho(k,k) * rhoVar );
        }

    }

    // modify useful quantities, only rhoU impacted
    //recompute rhoU as the rhos have changed
    RhoU = createRhoU( U, SigmaRho );

    logP_SigmaRho = logP;
}

/*
double BVS_dMVP::logPSigmaRho(
    const arma::mat& SigmaRho,
    const unsigned int N,
    const double nu)
{
    double logP = 0.;

    unsigned int L = SigmaRho.n_cols;

    double thisSigmaTT;
    arma::uvec conditioninIndexes;
    unsigned int nConditioninIndexes;

    double a,b;
    arma::mat rhoVar; // inverse matrix of the residual elements of Sigma in the component
    arma::rowvec rhoMean; // this is the partial Schur complement, needed for the sampler

    arma::uvec singleIdx_k(1); // needed for convention with arma::submat
    for( unsigned int k=0; k<L; ++k )
    {
        singleIdx_k(0) = k;

        // start computing interesting things
        thisSigmaTT = 1.0;// psi; //Sigma(k,k);

        nConditioninIndexes = k;
        conditioninIndexes.zeros( nConditioninIndexes );

        if( nConditioninIndexes > 0 )
        {
            conditioninIndexes = arma::regspace<arma::uvec>(0, k-1);
            rhoVar = arma::eye(nConditioninIndexes,nConditioninIndexes);
            rhoMean = arma::zeros<arma::rowvec>(nConditioninIndexes); // off-diagonals are zeros
        }

        // *** Diagonal Element

        // Compute parameters
        // a = 0.5 * ( N + nu - L + nConditioninIndexes + 1. ) ;
        // a = 0.5 * ( nu - L + 2*nConditioninIndexes + 1. ) ;
        a = 0.5 * (double)N + (double)k;
        b = 0.5 * thisSigmaTT ;

        logP += BVS_subfunc::logPDFIGamma( SigmaRho(k,k), a, b );


        // *** Off-Diagonal Element(s)
        if( nConditioninIndexes > 0 )
        {
            logP += BVS_subfunc::logPDFNormal( SigmaRho( conditioninIndexes, singleIdx_k ), rhoMean.t(), SigmaRho(k,k) * rhoVar );
        }

    }

    return logP;
}
*/

arma::mat BVS_dMVP::createRhoU(
    const arma::mat& U,
    const arma::mat&  SigmaRho)
{

    unsigned int N = U.n_rows;
    unsigned int L = U.n_cols;

    arma::mat RhoU = arma::zeros<arma::mat>(N, L);

    for( unsigned int k=1; k < L; ++k)
    {
        for(unsigned int l=0 ; l<k ; ++l)
        {
            if(  SigmaRho(k,l) != 0 )
                RhoU.col(k) += U.col(l) *  SigmaRho(k,l);
        }
    }

    return RhoU;
}


/*
void BVS_dMVP::sampleZ(
    arma::mat& Z,
    const arma::mat& betas,
    const DataClass &dataclass
)
{
    unsigned int N = dataclass.y.n_rows;
    unsigned int L = dataclass.y.n_cols;

    // Z.elem(arma::find( (dataclass.y == 1. && externalZ < 0.) || (dataclass.y == 0. && externalZ > 0.) )).fill(0.);
    Z.fill(0.); // reset all entries
    arma::mat Mus = dataclass.X * betas;

    for( unsigned int k=0; k<L; ++k)
    {
        arma::uvec singleIdx_k = {k};
        // arma::uvec VS_IN_k = arma::find(betas.col(k) != 0.); // include intercept

        for( unsigned int i=0; i<N; ++i)
        {
            // double mu_ik = arma::as_scalar( dataclass.X.cols(VS_IN_k) * betas(VS_IN_k, singleIdx_k) );

            if( dataclass.y(i,k) && (Mus(i,k) > 0.) )
            {
                Z(i, k) = BVS_subfunc::randTruncNorm( Mus(i,k), 1., 0., 1.0e+6 );
            }
            else if( (dataclass.y(i,k) == 0.) && (Mus(i,k) < 0.) )
            {
                Z(i, k) = BVS_subfunc::randTruncNorm( Mus(i,k), 1., -1.0e+6, 0. );
            }
        }

    }

}
*/

// update both Z and D
void BVS_dMVP::sampleZ(
    arma::mat& mutantZ,
    arma::mat& mutantD,
    const arma::mat& betas,
    const arma::mat& Psi,
    const DataClass &dataclass
)
{
    unsigned int N = mutantZ.n_rows;
    unsigned int L = mutantZ.n_cols;

    // arma::mat Psi = arma::zeros<arma::mat>(L, L);
    // updatePsi( SigmaRho, Psi );
    arma::vec dinv = 1.0 / arma::sqrt(Psi.diag());
    arma::mat Dinv = arma::diagmat(dinv);

    // std::cout << "...Dinv=\n" << Dinv << "\n";
    arma::mat RR = Dinv * Psi * Dinv;
    RR = 0.5 * (RR + RR.t()); // enforce symmetry numerically
    arma::mat Rinv;
    if( !arma::inv_sympd( Rinv, RR ) )
    {
        arma::inv(Rinv, RR, arma::inv_opts::allow_approx);
    }

    // transform back to original formulation of the latent variable
    arma::mat Z0 = mutantZ * Dinv;
    arma::mat Mus = dataclass.X * betas * Dinv;

    arma::mat Z = arma::zeros<arma::mat>(N, L); // reset all entries
    arma::uvec singleIdx_k;
    arma::uvec all_idx = arma::linspace<arma::uvec>(0, L-1, L);


    for( unsigned int k=0; k<L; ++k)
    {
        // Z.col(k) = zbinprobit( dataclass.y.col(k), Mus.col(k) );

        arma::uvec singleIdx_k = {k};
        arma::uvec excludeIdx_k = all_idx;
        excludeIdx_k.shed_row(k);

        arma::mat Rmm = RR.submat(excludeIdx_k, excludeIdx_k);
        arma::mat chol_Rmm;
        if(!arma::chol(chol_Rmm, Rmm, "lower"))
        {
            arma::mat Rmm_jit = Rmm + 1.0e-10 * arma::eye<arma::mat>(Rmm.n_rows, Rmm.n_cols);
            if (!arma::chol(chol_Rmm, Rmm_jit, "lower"))
            {
                throw std::runtime_error("Cholesky failed for R_{-k,-k}");
            }
        }

        arma::rowvec Rkm = RR.submat(singleIdx_k, excludeIdx_k);
        arma::vec    Rmk = RR.submat(excludeIdx_k, singleIdx_k);
        // Solve R_{-k,-k}^{-1} R_{-k,k}
        arma::vec sol2 = arma::solve(arma::trimatl(chol_Rmm), Rmk, arma::solve_opts::fast);
        sol2 = arma::solve(arma::trimatu(chol_Rmm.t()), sol2, arma::solve_opts::fast);

        double var_k = 1.0 - arma::as_scalar(Rkm * sol2);
        var_k = std::max(1.0e-16, var_k);
        double sd_k  = std::sqrt(var_k);

        for( unsigned int i=0; i<N; ++i)
        {
            arma::uvec singleIdx_i = {i};
            arma::vec rhs = (Z0.submat(singleIdx_i, excludeIdx_k).t() - Mus.submat(singleIdx_i, excludeIdx_k).t());
            arma::vec sol = arma::solve(arma::trimatl(chol_Rmm), rhs, arma::solve_opts::fast);
            sol = arma::solve(arma::trimatu(chol_Rmm.t()), sol, arma::solve_opts::fast);

            double mu_ik = Mus(i,k) + arma::as_scalar( Rkm * sol );
            /*
            if( dataclass.y(i, k) == 1. && mu_ik > 0.)
            {
                Z(i, k) = BVS_subfunc::randTruncNorm( mu_ik, sigmaZ_ik, 0., 1.0e+6);
            }
            else if( dataclass.y(i, k) == 0. && mu_ik < 0.)
            {
                Z(i, k) = BVS_subfunc::randTruncNorm( mu_ik, sigmaZ_ik, -1.0e+6, 0. );
            }
            */
            Z(i,k) = zbinprobit(dataclass.y(i, k), mu_ik, sd_k);

            // }
        }

        // if(Rinv(k, k) <= 0 )
        // {
        //     throw std::runtime_error(" Negative parameter in sampleZ Rinv(k, k)");
        // }

        mutantD(k, k) = std::sqrt( BVS_subfunc::randIGamma( (double)(L + 1)/2.0, Rinv(k, k)/2.0 ) );
        // mutantD(k, k) = D(k, k);
        // std::cout << "...k=" << k << "; mutantD(k, k)=" << mutantD(k, k) << "; Rinv(k, k)=" << Rinv(k, k)  << "\n";
    }

    // transform to Z in the reparametrized space
    mutantZ = Z * mutantD;
}

arma::vec BVS_dMVP::zbinprobit(
    const arma::vec& y,
    const arma::vec& m
)
{
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

double BVS_dMVP::zbinprobit(
    const double y,
    const double m,
    const double sigma
)
{

    double u = R::runif(0., 1.);
    double cd = arma::normcdf( -m / sigma );

    if(cd > 1.0-lowerbound0)
    {
        cd = 1.0-lowerbound0;
    }
    else if(cd < lowerbound0)
    {
        cd = lowerbound0;
    }

    double pu = (u * cd) * (1.0 - 2.0 * y) + (u + cd) * y;
    double z = m + sigma * R::qnorm(pu, 0., 1., true, false);

    return z;
}


void BVS_dMVP::updatePsi(
    const arma::mat& SigmaRho,
    arma::mat& Psi
)
{
    unsigned int L = SigmaRho.n_rows;
    Psi.zeros(L, L); // RESET THE WHOLE MATRIX !!!
    /*
    Psi(0, 0) = SigmaRho(0, 0);
    Psi(1, 0) = SigmaRho(1, 0) * Psi(0, 0);
    Psi(0, 1) = Psi(1, 0);
    Psi(1, 1) = SigmaRho(1, 1);
    Psi(1, 1) += SigmaRho(1, 0) * SigmaRho(1, 0) * Psi(0, 0);

    Psi(2, 0) += SigmaRho(2, 0) * Psi(0, 0);
    Psi(2, 0) += SigmaRho(2, 1) * Psi(1, 0);
    Psi(0, 2) = Psi(2, 0);

    Psi(2, 1) += SigmaRho(2, 1) * Psi(1, 1);
    Psi(2, 1) += SigmaRho(2, 0) * Psi(0, 1);
    Psi(1, 2) = Psi(2, 1);

    Psi(2, 2) = SigmaRho(2, 2);
    Psi(2, 2) += SigmaRho(2, 0) * SigmaRho(2, 0) * Psi(0, 0) +
                 SigmaRho(2, 1) * SigmaRho(2, 1) * Psi(1, 1) +
                 2.0 * SigmaRho(2, 0) * SigmaRho(2, 1) * Psi(1, 0);

    for ( unsigned int j=3; j<L; ++j)
    {
        for ( unsigned int i=0; i<j; ++i )
        {
            Psi(j, i) = 0.; // initializing to make sure it

            Psi(j, i) += SigmaRho(j, i) * Psi(i, i);
            for ( unsigned int k=0; k<j; ++k )
            {
                Psi(j, i) += SigmaRho(j, k) * Psi(k, i);
            }
            Psi(j, i) -= SigmaRho(j, i) * Psi(i, i); // subtract the term for k=i: SigmaRho(j, k) * Psi(k, i)
            Psi(i, j) = Psi(j, i);
        }

        // Second compute Psi(j,j):
        Psi(j, j) += SigmaRho(j, j);
        for ( unsigned int k=0; k<j; ++k )
        {
            Psi(j, j) += SigmaRho(j, k) * SigmaRho(j, k) * Psi(k, k);
        }
        for ( unsigned int l=1; l<j; ++l )
        {
            for ( unsigned int k=0; k<l; ++k )
            {
                Psi(j, j) += 2.0 * SigmaRho(j, k) * SigmaRho(j, l) * Psi(l, k);
            }
        }
    }
    */
    Psi(0, 0) = SigmaRho(0, 0);

    for (unsigned int j = 1; j < L; ++j)
    {
        auto Psi_sub = Psi.submat(0, 0, j - 1, j - 1);
        auto v = SigmaRho.submat(j, 0, j, j - 1);

        // off-diagonals
        arma::rowvec s = v * Psi_sub;
        Psi.submat(j, 0, j, j - 1) = s;
        Psi.submat(0, j, j - 1, j) = s.t();

        // diagonal
        double diag = SigmaRho(j, j) + arma::as_scalar(s * v.t());
        Psi(j, j) = diag;
    }

    // std::cout << "updatePsi(): Psi=\n" << Psi << "\n";
    //Psi += arma::eye(nOutcomes, nOutcomes) * 10.0;
    //if (!Psi.is_symmetric())
    //    std::cout << "\n...Debug -- Psi is not symmetric!!!" << std::endl;
    if (!Psi.is_sympd())
    {
        approx_sympd(Psi);
    }
    // generate symmetric matrix by reflecting the lower triangle to the upper triangle
    // Psi = arma::symmatl(Psi);

    // return Psi;
}

void BVS_dMVP::approx_sympd(arma::mat& x)
{
    /*
        std::cout << "\n...Debug -- Psi is not positive definite!!!" << std::endl;
        Psi.save("Psi.txt",arma::raw_ascii);
        SigmaRho.save("sigmaRho.txt",arma::raw_ascii);

        arma::mat Psi0, invPsi;
        arma::inv(invPsi, Psi, arma::inv_opts::allow_approx);
        arma::inv(Psi0, invPsi, arma::inv_opts::allow_approx);
         */

    // find nearest positive definite matrix
    // ref: https://stackoverflow.com/questions/61639182/find-the-nearest-postive-definte-matrix-with-eigen
    // the converted matrix can be the same til 12 digits:) It might mean that NPD is just due to numerical issue
    Eigen::MatrixXd eigen_x = Eigen::Map<Eigen::MatrixXd>(x.memptr(), x.n_rows, x.n_cols);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(0.5 * (eigen_x + eigen_x.transpose()));
    Eigen::MatrixXd A = solver.eigenvectors() * solver.eigenvalues().cwiseMax(0).asDiagonal() * solver.eigenvectors().transpose();
    //arma::mat x0  = matrixxd_to_armamat(A);
    x = arma::mat(A.data(), A.rows(), A.cols(), true, false);
    x = 0.5 * (x + x.t()); // to be sure for symmetry
    /*
    if (!Psi0.is_sympd())
    {
        std::cout << "\n...Debug -- Psi0 is not positive definite!!!" << std::endl;
        Psi0.save("Psi0.txt",arma::raw_ascii);
    }
    Psi = Psi0;
    */

    Rcpp::Rcout << "updatePsi()-Eigen: Psi=\n" << x << "\n";
}
