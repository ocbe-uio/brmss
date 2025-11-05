/* not yet implemented*/

#include "simple_gibbs.h"
#include "BVS_subfunc.h"
#include "BVS_iMVP.h"

void BVS_iMVP::mcmc(
    unsigned int nIter,
    unsigned int burnin,
    unsigned int thin,
    // double& tau0Sq, // no need tau0Sq due to flat prior for beta0
    // arma::vec& tauSq,
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

    arma::mat Z = arma::zeros<arma::mat>(N, L);
    arma::mat logP_gamma = arma::zeros<arma::mat>(p, L);
    // arma::vec sigmaSq(L, arma::fill::value(1.0));

    // std::cout << "...debug12\n";
    gamma_acc_count = 0;
    for(unsigned int l=0; l<L; ++l)
    {
        double pi = R::rbeta(hyperpar.piA, hyperpar.piB);

        for(unsigned int j=1; j<p; ++j) // starting from j=1 due to intercept
        {
            gammas(j, l) = R::rbinom(1, pi);
            logP_gamma(j, l) = BVS_subfunc::logPDFBernoulli(gammas(j, l), pi);

        }

        // initialize latent response variables
        for(unsigned int i=0; i<N; ++i)
        {
            if( dataclass.y(i,l) )
            {
                Z(i, l) = BVS_subfunc::randTruncNorm( 1., 1., 0., 1.0e+6 );
            }
            else
            {
                Z(i, l) = BVS_subfunc::randTruncNorm( -1, 1., -1.0e+6, 0. );
            }
        }
    }

    double tauSq = 1.0;
    double logP_tau = BVS_subfunc::logPDFIGamma( tauSq, hyperpar.tauA, hyperpar.tauB );
    double log_likelihood = logLikelihood( Z, gammas, betas, tauSq, dataclass );

    // std::cout << "...debug13\n";
    arma::vec loglik = arma::zeros<arma::vec>(N);
    loglikelihood_conditional(
        Z,
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

        // std::cout << "...debug14\n";

        // update quantities based on the new betas
        sampleTau( tauSq, logP_tau, log_likelihood, Z, hyperpar, dataclass, gammas, betas);
        // /*testing*/tauSq = 1.0;


        // std::cout << "...debug16\n";
        // update \gammas -- variable selection indicators
        sampleGamma(
            gammas,
            gammaSampler,
            logP_gamma,
            gamma_acc_count,
            log_likelihood,
            hyperpar,
            betas,
            // tau0Sq,
            tauSq,
            Z,
            dataclass
        );

        // std::cout << "...debug18\n";
        betas.elem(arma::find(gammas == 0)).fill(0.);

        // update post-hoc betas
        gibbs_betas(
            betas,
            gammas,
            // sigmaSq,
            // tau0Sq,
            tauSq,
            Z,
            dataclass
        );

        // update latent response variables
        sampleZ(Z, betas,  dataclass);

        log_likelihood = logLikelihood( Z, gammas, betas, tauSq, dataclass);

    // if(std::isnan(log_likelihood)){
    //     std::cout << "...main mcmc() Z=" << Z.t() <<
    //     "...gammas=" << gammas <<
    //     "...betas=" << betas << "\n";
    // }
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
            // sigmaSq_mcmc[1+nIter_thin_count] = sigmaSq[0];
            beta_mcmc.row(1+nIter_thin_count) = arma::vectorise(betas).t();
            tauSq_mcmc[1+nIter_thin_count] = tauSq;//hyperpar.tauSq; // TODO: only keep the firs one for now

            gamma_mcmc.row(1+nIter_thin_count) = arma::vectorise(gammas).t();

            loglikelihood_conditional(
                Z,
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

// joint likelihood f(Y,Z|...)
double BVS_iMVP::logLikelihood(
    const arma::mat& Z,
    const arma::umat& gammas,
    const arma::mat& betas,
    const double tauSq,
    const DataClass &dataclass)
{
    // dimensions
    unsigned int N = dataclass.X.n_rows;
    unsigned int L = dataclass.y.n_cols;

    double logP = 0.;

    //Option1: The following uses f(Y,Z|gammas) = f(Y|Z)f(Z|gammas)
    /*
    arma::mat normcdf_Z = arma::normcdf(Z);
    normcdf_Z.elem(arma::find(normcdf_Z > 1.0-lowerbound0)).fill(1.0-lowerbound0);
    normcdf_Z.elem(arma::find(normcdf_Z < lowerbound0)).fill(lowerbound0);

#ifdef _OPENMP
    #pragma omp parallel for default(shared) reduction(+:logP)
#endif

    for( unsigned int k=0; k<L; ++k)
    {
        arma::vec res = Z.col(k) - arma::mean( Z.col(k) );

        arma::uvec VS_IN_k = arma::find(gammas.col(k));
        VS_IN_k.shed_row(0); // exclude intercept

        arma::mat W_k;
        W_k = dataclass.X.cols(VS_IN_k).t() * dataclass.X.cols(VS_IN_k) + tauSq * arma::eye<arma::mat>(VS_IN_k.n_elem,VS_IN_k.n_elem);

        arma::mat invW_k;
        if( !arma::inv_sympd( invW_k, W_k ) )
        {
            arma::inv(invW_k, W_k, arma::inv_opts::allow_approx);
        }

        double S_gamma = arma::as_scalar( res.t()*res );
        S_gamma -= arma::as_scalar( res.t()*dataclass.X.cols(VS_IN_k) * invW_k * dataclass.X.cols(VS_IN_k).t() * res );

        double sign, tmp;
        arma::log_det(tmp, sign, invW_k );
        logP += 0.5*tmp;

        logP -= 0.5 * (double)VS_IN_k.n_elem * log(tauSq);

        logP -= 0.5*((double)N - 1.0) * std::log(S_gamma);

        // add probit log-likelihood log[f(Y|Z)]
        
        logP += arma::sum( dataclass.y.col(k) % arma::log(normcdf_Z.col(k)) +
                           (1.0-dataclass.y.col(k)) % arma::log(1.0-normcdf_Z.col(k)) );
        
    }
    */

    //Option2: The following uses f(Y,Z|betas) = f(Y|Z)f(Z|betas)
    /*
    arma::mat Z_tmp = Z;
    sampleZ(Z_tmp, betas,  dataclass);
    arma::vec loglik_tmp;
    loglikelihood_conditional(Z_tmp, dataclass, loglik_tmp);
    // logP += arma::accu(arma::log(arma::normcdf(Z_tmp)) );
    logP += arma::sum( loglik_tmp );
    */

    //Option3: The following uses log f(y|b,x) = y*log(Φ(xb)) + (1-y)*log(1-Φ(xb))
    
    // arma::mat XB = dataclass.X * betas;
    // logP = arma::accu( dataclass.y % arma::log(arma::normcdf(XB)) +
    //                (1.0-dataclass.y) % arma::log(1.0-arma::normcdf(XB)) );

    
    arma::mat normcdf_Z = arma::normcdf(dataclass.X * betas);
    normcdf_Z.elem(arma::find(normcdf_Z > 1.0-lowerbound0)).fill(1.0-lowerbound0);
    normcdf_Z.elem(arma::find(normcdf_Z < lowerbound0)).fill(lowerbound0);
    logP = arma::accu( dataclass.y % arma::log(normcdf_Z) +
                   (1.0-dataclass.y) % arma::log(1.0-normcdf_Z) );
    
    if(std::isnan(logP)){
        ::Rf_error("...logLikelihood() std::isnan(logP) cdf(XB): (min,max)="); 
        //normcdf_Z.min() << ", " << normcdf_Z.max() << "; sum(B)=" << arma::accu(betas) << "\n";
    }

    return logP; // this is un-normalized log-likelihood, different from R-pkg BayesSUR

}

// individual loglikelihoods f(Y|Z)
void BVS_iMVP::loglikelihood_conditional(
    const arma::mat& Z,
    const DataClass &dataclass,
    arma::vec& loglik)
{

    // loglik.zeros(N); // RESET THE WHOLE VECTOR !!!

    // if(Z.has_nan()){
    //     std::cout << "...loglikelihood_conditional() Z=\n" << Z;
    // }
    loglik =
        arma::sum( dataclass.y % arma::log(arma::normcdf(Z))  +
                   (1.0-dataclass.y) % arma::log(1.0-arma::normcdf(Z)), 1 );

}


void BVS_iMVP::gibbs_betas(
    arma::mat& betas,
    const arma::umat& gammas,
    // const arma::vec& sigmaSq,
    // const double tau0Sq,
    const double tauSq,
    const arma::mat& Z,
    const DataClass &dataclass)
{

    // unsigned int N = dataclass.X.n_rows;
    // unsigned int p = dataclass.X.n_cols;
    unsigned int L = Z.n_cols;

    // TODO: yMean is needed if y is not standardized

    for(unsigned int k=0; k<L; ++k)
    {

        arma::uvec VS_IN_k = arma::find(gammas.col(k)); // include intercept

        arma::vec diag_elements = arma::vec(VS_IN_k.n_elem, arma::fill::value(1./tauSq));
        diag_elements[0] = 1.;// /tau0Sq;

        arma::mat invW = dataclass.X.cols(VS_IN_k).t() * dataclass.X.cols(VS_IN_k) + arma::diagmat(diag_elements);
        arma::mat W;
        if( !arma::inv_sympd( W,  invW ) )
        {
            arma::inv(W, invW, arma::inv_opts::allow_approx);
        }

        arma::vec mu = W * dataclass.X.cols(VS_IN_k).t() * Z.col(k);
        arma::vec beta_mask = BVS_subfunc::randMvNormal( mu, W );

        arma::uvec singleIdx_k = {k};
        betas(VS_IN_k, singleIdx_k) = beta_mask;
    }

}

void BVS_iMVP::gibbs_betaK(
    const unsigned int k,
    arma::mat& betas,
    const arma::umat& gammas,
    // const double tau0Sq,
    const double tauSq,
    const arma::mat& Z,
    const DataClass &dataclass)
{

    // unsigned int N = dataclass.X.n_rows;
    // unsigned int p = dataclass.X.n_cols;
    unsigned int L = Z.n_cols;

    arma::uvec VS_IN_k = arma::find(gammas.col(k)); // include intercept

    arma::vec diag_elements = arma::vec(VS_IN_k.n_elem, arma::fill::value(1./tauSq));
    diag_elements[0] = 1.;// /tau0Sq;

    arma::mat invW = dataclass.X.cols(VS_IN_k).t() * dataclass.X.cols(VS_IN_k) + arma::diagmat(diag_elements);
    arma::mat W;
    if( !arma::inv_sympd( W,  invW ) )
    {
        arma::inv(W, invW, arma::inv_opts::allow_approx);
    }

    arma::vec mu = W * dataclass.X.cols(VS_IN_k).t() * Z.col(k);
    arma::vec beta_mask = BVS_subfunc::randMvNormal( mu, W );

    arma::uvec singleIdx_k = {k};
    betas(VS_IN_k, singleIdx_k) = beta_mask;

}


void BVS_iMVP::sampleGamma(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    double& log_likelihood,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    // const double tau0Sq,
    const double tauSq,

    arma::mat& Z,
    const DataClass &dataclass)
{

    // std::cout << "...debug21\n";
    arma::umat proposedGamma = gammas; // copy the original gammas and later change the address of the copied one
    arma::mat proposedGammaPrior;
    arma::uvec updateIdx;

    double logProposalRatio = 0;

    unsigned int N = Z.n_rows;
    unsigned int p = gammas.n_rows - 1;
    unsigned int L = gammas.n_cols;

    // define static variables for global updates for the use of bandit algorithm
    // initial value 0.5 here forces shrinkage toward 0 or 1
    static arma::mat banditAlpha = arma::mat(p, L, arma::fill::value(0.5));
    static arma::mat banditBeta = arma::mat(p, L, arma::fill::value(0.5));

    // std::cout << "...debug22\n";
    // decide on one component
    unsigned int componentUpdateIdx = 0;
    if (L > 1)
    {
        componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    }
    arma::uvec singleIdx_k = { componentUpdateIdx };

    // std::cout << "...debug23\n";
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

    // std::cout << "...debug24\n";
    // note only one outcome is updated
    // update log probabilities

    // compute logProposalGammaRatio, i.e. proposedGammaPrior - logP_gamma
    double logProposalGammaRatio = 0.;

    proposedGammaPrior = logP_gamma; // copy the original one and later change the address of the copied one

    for(auto i: updateIdx)
    {
        // double pi = R::rbeta(hyperpar.piA + (double)(proposedGamma(1+i,componentUpdateIdx)),
        //                      hyperpar.piB + (double)(p) - (double)(proposedGamma(1+i,componentUpdateIdx)));
        double pi = R::rbeta(hyperpar.piA + (double)(arma::sum(gammas.row(1+i))),
                             hyperpar.piB + (double)L - (double)(arma::sum(gammas.row(1+i))));
        proposedGammaPrior(1+i,componentUpdateIdx) = BVS_subfunc::logPDFBernoulli( proposedGamma(1+i,componentUpdateIdx), pi );
        logProposalGammaRatio +=  proposedGammaPrior(1+i, componentUpdateIdx) - logP_gamma(1+i, componentUpdateIdx);
    }

    // std::cout << "...debug25\n";
    arma::mat proposedBeta = betas;
    proposedBeta.elem(arma::find(proposedGamma == 0)).fill(0.);

    gibbs_betaK(
        componentUpdateIdx,
        proposedBeta,
        proposedGamma,
        // tau0Sq,
        tauSq,
        Z,
        dataclass
    );

    // std::cout << "...debug26\n";
    // TODO: Do we need proposedZ for proposedLikelihood?
    arma::mat proposedZ = Z;
    // sampleZ(proposedZ, proposedBeta,  dataclass);

    // compute logLikelihoodRatio
    double proposedLikelihood = logLikelihood( proposedZ, proposedGamma, proposedBeta, tauSq, dataclass );

    // std::cout << "...debug27\n";
    double logLikelihoodRatio = proposedLikelihood - log_likelihood;

    // Here we need always compute the proposal and original ratios, in particular the likelihood, since betas are updated
    double logAccProb = logProposalGammaRatio +
                        logLikelihoodRatio +
                        logProposalRatio;

    // std::cout << "...debug logAccProb=" << logAccProb << 
    // "; proposedLikelihood=" << proposedLikelihood << 
    // "; log_likelihood=" << log_likelihood << 
    // "; logProposalGammaRatio=" << logProposalGammaRatio << 
    // "; logProposalRatio=" << logProposalRatio << "\n";

    if( std::log(R::runif(0,1)) < logAccProb )
    {
        gammas = proposedGamma;
        logP_gamma = proposedGammaPrior;
        log_likelihood = proposedLikelihood;
        Z = proposedZ;
        betas = proposedBeta;

        ++gamma_acc_count;
    }

    // std::cout << "...debug28\n";
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

// random walk MH sampler; no full conditional due to integrated out beta
void BVS_iMVP::sampleTau(
    double& tauSq,
    double& logP_tau,
    double& log_likelihood,
    const arma::mat& Z,
    const hyperparClass& hyperpar,
    const DataClass& dataclass,
    const arma::umat& gammas, // TODO: this is not needed
    const arma::mat& betas
)
{
    double var_tau_proposal = 1.0; //2.38;
    double proposedTau = std::exp( std::log(tauSq) + R::rnorm(0.0, var_tau_proposal) );

    // double proposedTauPrior = logPTau( proposedTau );
    double proposedTauPrior = BVS_subfunc::logPDFIGamma( proposedTau, hyperpar.tauA, hyperpar.tauB );
    double proposedLikelihood = logLikelihood( Z, gammas, betas, proposedTau, dataclass );

    double logAccProb = (proposedTauPrior + proposedLikelihood) - (logP_tau + log_likelihood);

    if( std::log(R::runif(0,1)) < logAccProb )
    {
        tauSq = proposedTau;
        logP_tau = proposedTauPrior;
        log_likelihood = proposedLikelihood;

        // ++tau_acc_count;
    }
}

/*
void BVS_iMVP::sampleZ(
    arma::mat& Z,
    const arma::mat& betas,
    // const arma::umat& gammas,
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
        // arma::uvec VS_IN_k = arma::find(gammas.col(k)); // include intercept

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

void BVS_iMVP::sampleZ(
    arma::mat& Z,
    const arma::mat& betas,
    // const arma::umat& gammas,
    const DataClass &dataclass
)
{
    unsigned int N = dataclass.y.n_rows;
    unsigned int L = dataclass.y.n_cols;

    Z.fill(0.); // reset all entries
    arma::mat Mus = dataclass.X * betas;

    for( unsigned int k=0; k<L; ++k)
    {
        Z.col(k) = zbinprobit( dataclass.y.col(k), Mus.col(k) );
    }

}

arma::vec BVS_iMVP::zbinprobit(
    const arma::vec& y,
    const arma::vec& m
)
{
    unsigned int n = m.n_elem;

    // the following code is translated from R code in  https://en.wikipedia.org/wiki/Probit_model
    /*
    zbinprobit <- function(y, X, beta, n) {
        meanv <- X %*% beta
        u <- runif(n)  # uniform(0,1) random variates
        cd <- pnorm(-meanv)  # cumulative normal CDF
        pu <- (u * cd) * (1 - 2 * y) + (u + cd) * y
        cpui <- qnorm(pu)  # inverse normal CDF
        z <- meanv + cpui  # latent vector
        return(z)
    }
    */
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
