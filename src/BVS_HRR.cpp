/* Log-likelihood for the use in Metropolis-Hastings sampler*/

#include "simple_gibbs.h"
#include "BVS_subfunc.h"
#include "BVS_HRR.h"

void BVS_HRR::mcmc(
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

    arma::mat logP_gamma = arma::zeros<arma::mat>(p, L);; // this is declared to be updated in the M-H sampler for gammas
    arma::vec sigmaSq(L, arma::fill::value(1.0));

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
    loglikelihood_conditional(
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
            Rcpp::Rcout << "\r[" <<                                           //'\r' aka carriage return should move printer's cursor back at the beginning of the current line
                        std::string(cTotalLength * (m + 1.) / nIter, '#') <<        // printing filled part
                        std::string(cTotalLength * (1. - (m + 1.) / nIter), '-') << // printing empty part
                        "] " << (int)((m + 1.) / nIter * 100.0) << "%\r";             // printing percentage

        // std::cout << "...debug14\n";
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

        // std::cout << "...debug15\n";
        // update Gaussian's shape parameter
        gibbs_sigmaSq(
            sigmaSq,
            hyperpar.sigmaA,
            hyperpar.sigmaB,
            dataclass,
            betas
        );

        // std::cout << "...debug16\n";
        // update \gammas -- variable selection indicators
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

        // std::cout << "...debug18\n";
        betas.elem(arma::find(gammas == 0)).fill(0.); 
        
        // update post-hoc betas
        gibbs_betas(
            betas,
            gammas,
            sigmaSq,
            tau0Sq,
            tauSq,
            dataclass
        );

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
            sigmaSq_mcmc[1+nIter_thin_count] = sigmaSq[0];
            beta_mcmc.row(1+nIter_thin_count) = arma::vectorise(betas).t();
            tauSq_mcmc[1+nIter_thin_count] = tauSq[0];//hyperpar.tauSq; // TODO: only keep the firs one for now

            gamma_mcmc.row(1+nIter_thin_count) = arma::vectorise(gammas).t();

            loglikelihood_conditional(
                betas,
                sigmaSq,
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

double BVS_HRR::loglikelihood(
    const arma::umat& gammas,
    const arma::vec& tauSq,
    double sigmaA,
    double sigmaB,
    const DataClass &dataclass)
{
    // dimensions
    unsigned int N = dataclass.X.n_rows;
    unsigned int L = dataclass.y.n_cols;

    double logP = 0.;

#ifdef _OPENMP
    #pragma omp parallel for default(shared) reduction(+:logP)
#endif

    for( unsigned int k=0; k<L; ++k)
    {
        // arma::uvec singleIdx_k = { k };
        arma::vec res = dataclass.y.col(k) - arma::mean( dataclass.y.col(k) );

        arma::uvec VS_IN_k = arma::find(gammas.col(k));
        VS_IN_k.shed_row(0); // exclude intercept

        arma::mat W_k;
        W_k = dataclass.X.cols(VS_IN_k).t() * dataclass.X.cols(VS_IN_k) + 1./tauSq[k] * arma::eye<arma::mat>(VS_IN_k.n_elem,VS_IN_k.n_elem);

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

        logP -= 0.5 * (double)VS_IN_k.n_elem * log(tauSq[k]);

        logP -= 0.5*(2.0*sigmaA + (double)N - 1.0) * std::log(2.0*sigmaB + S_gamma);
    }

    return logP; // this is un-normalized log-likelihood, different from R-pkg BayesSUR

}

// individual loglikelihoods
void BVS_HRR::loglikelihood_conditional(
    const arma::mat& betas,
    const arma::vec& sigmaSq,
    const DataClass &dataclass,
    arma::vec& loglik)
{
    // dimensions
    unsigned int N = dataclass.y.n_rows;
    unsigned int L = dataclass.y.n_cols;

    loglik.zeros(N); // RESET THE WHOLE VECTOR !!!

#ifdef _OPENMP
    #pragma omp parallel for
#endif

    for(unsigned int l=0; l<L; ++l)
    {
        arma::vec res = dataclass.y.col(l) - dataclass.X * betas.col(l);
        loglik += -0.5*log(2.*M_PI*sigmaSq[l]) -0.5/sigmaSq[l]* (res % res);

    }

}

/*
void BVS_HRR::gibbs_betaK(
    unsigned int componentUpdateIdx,
    arma::mat& betas,
    const arma::umat& gammas,
    const double tau0Sq,
    const double tauSq,
    const DataClass &dataclass)
{

    double logP = 0.;

    unsigned int N = dataclass.X.n_rows;
    unsigned int p = dataclass.X.n_cols;
    unsigned int L = dataclass.y.n_cols;

    // TODO: yMean is needed if y is not standardized

    arma::uvec VS_IN_k = arma::find(gammas.col(componentUpdateIdx));

    arma::vec diag_elements = arma::vec(VS_IN_k.n_elem, arma::fill::value(1./tauSq));
    diag_elements[0] = 1./tau0Sq;

    arma::mat invW = dataclass.X.cols(VS_IN_k).t() * dataclass.X.cols(VS_IN_k) / sigmaSq + arma::diagmat(diag_elements);
    arma::mat W;
    if( !arma::inv_sympd( W,  invW ) )
    {
        arma::inv(W, invW, arma::inv_opts::allow_approx);
    }

    arma::vec mu = W * dataclass.X.cols(VS_IN_k).t() * dataclass.y / sigmaSq;
    arma::vec beta_mask = BVS_subfunc::randMvNormal( mu, W );

    arma::uvec singleIdx_k = {componentUpdateIdx};
    betaK(VS_idx, singleIdx_k) = beta_mask;

}
    */

void BVS_HRR::gibbs_betas(
    arma::mat& betas,
    const arma::umat& gammas,
    const arma::vec& sigmaSq,
    const double tau0Sq,
    const arma::vec& tauSq,
    const DataClass &dataclass)
{

    // unsigned int N = dataclass.X.n_rows;
    // unsigned int p = dataclass.X.n_cols;
    unsigned int L = dataclass.y.n_cols;

    // TODO: yMean is needed if y is not standardized

    for(unsigned int k=0; k<L; ++k)
    {

        arma::uvec VS_IN_k = arma::find(gammas.col(k)); // include intercept

        arma::vec diag_elements = arma::vec(VS_IN_k.n_elem, arma::fill::value(1./tauSq[k]));
        diag_elements[0] = 1./tau0Sq;

        arma::mat invW = dataclass.X.cols(VS_IN_k).t() * dataclass.X.cols(VS_IN_k) / sigmaSq[k] + arma::diagmat(diag_elements);
        arma::mat W;
        if( !arma::inv_sympd( W,  invW ) )
        {
            arma::inv(W, invW, arma::inv_opts::allow_approx);
        }

        arma::vec mu = W * dataclass.X.cols(VS_IN_k).t() * dataclass.y.col(k) / sigmaSq[k];
        arma::vec beta_mask = BVS_subfunc::randMvNormal( mu, W );

        arma::uvec singleIdx_k = {k};
        betas(VS_IN_k, singleIdx_k) = beta_mask;
    }

}


void BVS_HRR::sampleGamma(
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


    // compute logLikelihoodRatio, i.e. proposedLikelihood - loglik
    // arma::vec proposedLikelihood = loglik;
    double loglik0 = loglikelihood( gammas, tauSq, hyperpar.sigmaA, hyperpar.sigmaB, dataclass );
    double proposedLikelihood = loglikelihood( proposedGamma, tauSq, hyperpar.sigmaA, hyperpar.sigmaB, dataclass );

    double logLikelihoodRatio = proposedLikelihood - loglik0;

    // Here we need always compute the proposal and original ratios, in particular the likelihood, since betas are updated
    double logAccProb = logProposalGammaRatio +
                        logLikelihoodRatio +
                        logProposalRatio;

    if( std::log(R::runif(0,1)) < logAccProb )
    {
        gammas = proposedGamma;
        logP_gamma = proposedGammaPrior;
        // loglik = proposedLikelihood;
        // betas = proposedBeta;

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


void BVS_HRR::gibbs_sigmaSq(
    arma::vec& sigmaSq,
    double a,
    double b,
    const DataClass& dataclass,
    const arma::mat& betas
)
{
    unsigned int N = dataclass.y.n_rows;
    unsigned int L = dataclass.y.n_cols;

    double a_sigma = a + 0.5 * (double)(N);

    for (unsigned int l=0; l<L; ++l)
    {
        double b_sigma = b + 0.5 * arma::as_scalar( 
            (dataclass.y.col(l) - dataclass.X*betas.col(l)).t() * 
            (dataclass.y.col(l) - dataclass.X*betas.col(l)) 
        );

        sigmaSq[l] = 1. / R::rgamma(a_sigma, 1. / b_sigma);
    }

}