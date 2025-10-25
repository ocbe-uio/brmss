/* Log-likelihood for the use in Metropolis-Hastings sampler*/

#include "BVS.h"
#include "arms_gibbs.h"

// TODO: make loglikelihood general for other response types
void BVS_Sampler::loglikelihood(
    const arma::mat& betas,
    double kappa,
    const DataClass &dataclass,
    arma::vec& loglik)
{
    // dimensions
    unsigned int N = dataclass.y.n_rows;
    unsigned int p = dataclass.X.n_cols;

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


void BVS_Sampler::sampleGamma(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    Family_Type familyType,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    arma::vec& loglik,
    const armsParmClass& armsPar,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    double kappa,
    double& tau0Sq,
    double& tauSq,

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
    if (L > 1)
    {
        componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    }
    arma::uvec singleIdx_k = { componentUpdateIdx };

    // Update the proposed Gamma with 'updateIdx' renewed via its address
    switch( gamma_sampler )
    {
    case Gamma_Sampler_Type::bandit:
        logProposalRatio += gammaBanditProposal( p, proposedGamma, gammas, updateIdx, componentUpdateIdx, banditAlpha );
        break;

    case Gamma_Sampler_Type::mc3:
        logProposalRatio += gammaMC3Proposal( p, proposedGamma, gammas, updateIdx, componentUpdateIdx );
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
        proposedGammaPrior(i,componentUpdateIdx) = logPDFBernoulli( proposedGamma(i,componentUpdateIdx), pi );
        logProposalGammaRatio +=  proposedGammaPrior(i, componentUpdateIdx) - logP_gamma(i, componentUpdateIdx);
    }

    arma::mat betas_proposal = betas;

    // update (addresses) 'betas_proposal' and 'logPosteriorBeta_proposal' based on 'proposedGamma'

    switch(familyType)
    {
    case Family_Type::weibull:
    {
        ARMS_Gibbs::arms_gibbs_beta_weibull(
            armsPar,
            hyperpar,
            betas_proposal,
            proposedGamma,
            tauSq,
            tau0Sq,
            kappa,
            dataclass
        );
        break;
    }

    case Family_Type::dirichlet:
    {
        double TOOD = 0.;
    }
    }


    // compute logLikelihoodRatio, i.e. proposedLikelihood - loglik
    arma::vec proposedLikelihood = loglik;
    loglikelihood( betas, kappa, dataclass, loglik );
    loglikelihood( betas_proposal, kappa, dataclass, proposedLikelihood );

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
        betas = betas_proposal;

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

void BVS_Sampler::sampleGammaProposalRatio(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    Family_Type familyType,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    arma::vec& loglik,
    const armsParmClass& armsPar,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    double kappa,
    double& tau0Sq,
    double& tauSq,

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
    if (L > 1)
    {
        componentUpdateIdx = static_cast<unsigned int>( R::runif( 0, L ) );
    }
    arma::uvec singleIdx_k = { componentUpdateIdx };

    // Update the proposed Gamma with 'updateIdx' renewed via its address
    switch( gamma_sampler )
    {
    case Gamma_Sampler_Type::bandit:
        logProposalRatio += gammaBanditProposal( p, proposedGamma, gammas, updateIdx, componentUpdateIdx, banditAlpha );
        break;

    case Gamma_Sampler_Type::mc3:
        logProposalRatio += gammaMC3Proposal( p, proposedGamma, gammas, updateIdx, componentUpdateIdx );
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
        proposedGammaPrior(i,componentUpdateIdx) = logPDFBernoulli( proposedGamma(i,componentUpdateIdx), pi );
        logProposalGammaRatio +=  proposedGammaPrior(i, componentUpdateIdx) - logP_gamma(i, componentUpdateIdx);
    }

    arma::mat betas_proposal = betas;

    // update (addresses) 'betas_proposal' and 'logPosteriorBeta_proposal' based on 'proposedGamma'
    double logPosteriorBeta, logPosteriorBeta_proposal;

    switch(familyType)
    {
    case Family_Type::weibull:
    {
        ARMS_Gibbs::arms_gibbs_beta_weibull(
            armsPar,
            hyperpar,
            betas_proposal,
            proposedGamma,
            tauSq,
            tau0Sq,
            kappa,
            dataclass
        );

        logPosteriorBeta = logPbeta(
            betas,
            tauSq,
            kappa,
            dataclass
        );
        logPosteriorBeta_proposal = logPbeta(
            betas_proposal,
            tauSq,
            kappa,
            dataclass
        );

        break;
    }

    case Family_Type::dirichlet:
    {
        double TOOD = 0.;
    }
    }

    double logPriorBetaRatio = logPDFNormal(betas_proposal, tauSq) - logPDFNormal(betas, tauSq);
    double logProposalBetaRatio = logPosteriorBeta - logPosteriorBeta_proposal;


    // compute logLikelihoodRatio, i.e. proposedLikelihood - loglik
    arma::vec proposedLikelihood = loglik;
    loglikelihood( betas, kappa, dataclass, loglik );
    loglikelihood( betas_proposal, kappa, dataclass, proposedLikelihood );

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
        betas = betas_proposal;

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

double BVS_Sampler::gammaMC3Proposal(
    unsigned int p,
    arma::umat& mutantGamma,
    const arma::umat gammas,
    arma::uvec& updateIdx,
    unsigned int componentUpdateIdx )
{
    //arma::umat mutantGamma = gammas;
    unsigned int n_updates_MC3 = std::max(5., std::ceil( (double)(p) / 5. )); //arbitrary number, should I use something different?

    Rcpp::IntegerVector entireIdx = Rcpp::seq( 0, p - 1);
    updateIdx = Rcpp::as<arma::uvec>(Rcpp::sample(entireIdx, n_updates_MC3, false)); // here 'replace = false'

    for( auto i : updateIdx)
    {
        mutantGamma(i,componentUpdateIdx) = ( R::runif(0,1) < 0.5 )? gammas(i,componentUpdateIdx) : 1-gammas(i,componentUpdateIdx); // could simply be ( 0.5 ? 1 : 0) ;
    }

    //return mutantGamma ;
    return 0. ; // pass this to the outside, it's the (symmetric) logProposalRatio
}

// sampler for proposed updates on gammas
double BVS_Sampler::gammaBanditProposal(
    unsigned int p,
    arma::umat& mutantGamma,
    const arma::umat gammas,
    arma::uvec& updateIdx,
    unsigned int componentUpdateIdx,
    arma::mat& banditAlpha )
{
    // define static variables for global updates
    // 'banditZeta' corresponds to pi in brmss
    static arma::vec banditZeta = arma::vec(p);

    static arma::vec mismatch = arma::vec(p);
    static arma::vec normalised_mismatch = arma::vec(p);
    static arma::vec normalised_mismatch_backwards = arma::vec(p);

    unsigned int n_updates_bandit = 4; // this needs to be low as its O(n_updates!)

    double logProposalRatio = 0.;

    for(unsigned int j=0; j<p; ++j)
    {
        // Sample Zs (only for relevant component)
        banditZeta(j) = R::rbeta(banditAlpha(j,componentUpdateIdx),banditAlpha(j,componentUpdateIdx));

        // Create mismatch (only for relevant outcome)
        mismatch(j) = (mutantGamma(j,componentUpdateIdx)==0)?(banditZeta(j)):(1.-banditZeta(j));   //mismatch
    }

    // normalised_mismatch = mismatch / arma::as_scalar(arma::sum(mismatch));
    normalised_mismatch = mismatch / arma::sum(mismatch);

    if( R::runif(0,1) < 0.5 )   // one deterministic update
    {
        // Decide which to update
        updateIdx = arma::zeros<arma::uvec>(1);
        //updateIdx(0) = randWeightedIndexSampleWithoutReplacement(p,normalised_mismatch); // sample the one
        updateIdx(0) = randWeightedIndexSampleWithoutReplacement(normalised_mismatch); // sample the one

        // Update
        mutantGamma(updateIdx(0),componentUpdateIdx) = 1 - gammas(updateIdx(0),componentUpdateIdx); // deterministic, just switch

        // Compute logProposalRatio probabilities
        normalised_mismatch_backwards = mismatch;
        normalised_mismatch_backwards(updateIdx(0)) = 1. - normalised_mismatch_backwards(updateIdx(0)) ;

        normalised_mismatch_backwards = normalised_mismatch_backwards / arma::sum(normalised_mismatch_backwards);

        logProposalRatio = ( std::log( normalised_mismatch_backwards(updateIdx(0)) ) ) -
                           ( std::log( normalised_mismatch(updateIdx(0)) ) );

    }
    else
    {
        // Decide which to update
        updateIdx = arma::zeros<arma::uvec>(n_updates_bandit);
        updateIdx = randWeightedIndexSampleWithoutReplacement(p,normalised_mismatch,n_updates_bandit); // sample n_updates_bandit indexes

        normalised_mismatch_backwards = mismatch; // copy for backward proposal

        // Update
        for(unsigned int i=0; i<n_updates_bandit; ++i)
        {
            // mutantGamma(updateIdx(i),componentUpdateIdx) = static_cast<unsigned int>(R::rbinom( 1, banditZeta(updateIdx(i)))); // random update
            unsigned int j = R::rbinom( 1, banditZeta(updateIdx(i))); // random update
            mutantGamma(updateIdx(i),componentUpdateIdx) = j;

            normalised_mismatch_backwards(updateIdx(i)) = 1.- normalised_mismatch_backwards(updateIdx(i));

            logProposalRatio += logPDFBernoulli(gammas(updateIdx(i),componentUpdateIdx),banditZeta(updateIdx(i))) -
                                logPDFBernoulli(mutantGamma(updateIdx(i),componentUpdateIdx),banditZeta(updateIdx(i)));
        }

        // Compute logProposalRatio probabilities
        normalised_mismatch_backwards = normalised_mismatch_backwards / arma::sum(normalised_mismatch_backwards);

        logProposalRatio += logPDFWeightedIndexSampleWithoutReplacement(normalised_mismatch_backwards,updateIdx) -
                            logPDFWeightedIndexSampleWithoutReplacement(normalised_mismatch,updateIdx);
    }

    return logProposalRatio; // pass this to the outside
    //return mutantGamma;

}


// subfunctions used for bandit proposal

arma::uvec BVS_Sampler::randWeightedIndexSampleWithoutReplacement(
    unsigned int populationSize,    // size of set sampling from
    const arma::vec& weights,       // (log) probability for each element
    unsigned int sampleSize         // size of each sample
) // sample is a zero-offset indices to selected items, output is the subsampled population.
{
    // note I can do everything in the log scale as the ordering won't change!
    arma::vec tmp = Rcpp::rexp( populationSize, 1. );
    arma::vec score = tmp - weights;
    arma::uvec result = arma::sort_index( score,"ascend" );

    return result.subvec(0,sampleSize-1);
}

// overload with sampleSize equal to one
unsigned int BVS_Sampler::randWeightedIndexSampleWithoutReplacement(
    const arma::vec& weights     // probability for each element
) // sample is a zero-offset indices to selected items, output is the subsampled population.
{
    // note I can do everything in the log scale as the ordering won't change!

    double u = R::runif(0,1);
    double tmp = weights(0);
    unsigned int t = 0;

    while(u > tmp)
    {
        // tmp = Utils::logspace_add(tmp,logWeights(++t));
        tmp += weights(++t);
    }

    return t;
}

// logPDF rand Weighted Indexes (need to implement the one for the original starting vector?)
double BVS_Sampler::logPDFWeightedIndexSampleWithoutReplacement(
    const arma::vec& weights,
    const arma::uvec& indexes
)
{
    // arma::vec logP_permutation = arma::zeros<arma::vec>((int)std::tgamma(indexes.n_elem+1));  //too big of a vector
    double logP_permutation = 0.;
    double tmp;

    std::vector<unsigned int> v = arma::conv_to<std::vector<unsigned int>>::from(arma::sort(indexes));
    // vector should be sorted at the beginning.

    arma::uvec current_permutation;
    arma::vec current_weights;

    do
    {
        current_permutation = arma::conv_to<arma::uvec>::from(v);
        current_weights = weights;
        tmp = 0.;

        while( current_permutation.n_elem > 0 )
        {
            tmp += log(current_weights(current_permutation(0)));
            current_permutation.shed_row(0);
            current_weights = current_weights/arma::sum(current_weights(current_permutation));   // this will gets array weights that do not sum to 1 in total, but will only use relevant elements
        }

        logP_permutation = logspace_add( logP_permutation,tmp );

    }
    while (std::next_permutation(v.begin(), v.end()));

    return logP_permutation;
}

double BVS_Sampler::logspace_add(
    double a,
    double b)
{

    if(a <= std::numeric_limits<float>::lowest())
        return b;
    if(b <= std::numeric_limits<float>::lowest())
        return a;
    return std::max(a, b) + std::log( (double)(1. + std::exp( (double)-std::abs((double)(a - b)) )));
}

double BVS_Sampler::logPDFBernoulli(unsigned int x, double pi)
{
    if( x > 1 ||  x < 0 )
        return -std::numeric_limits<double>::infinity();
    else
        return (double)(x) * std::log(pi) + (1.-(double)(x)) * std::log(1. - pi);
}

double BVS_Sampler::logPDFNormal(const arma::vec& x, const double& sigmaSq)  // zeroMean and independentVar
{
    unsigned int k = x.n_elem;
    double tmp = (double)k * std::log(sigmaSq); // log-determinant(Sigma)

    return -0.5*(double)k*log(2.*M_PI) -0.5*tmp - 0.5 * arma::as_scalar( x.t() * x ) / sigmaSq;

}

double BVS_Sampler::logPbeta(
           const arma::mat& betas,
           double tauSq,
           double kappa,
           const DataClass& dataclass)
{
    unsigned int p = dataclass.X.n_cols;

    arma::vec logMu = betas(0) + dataclass.X * betas.submat(1, 0, p, 0);
    // logMu.elem(arma::find(logMu > upperbound)).fill(upperbound);

    arma::vec lambdas = arma::exp(logMu) / std::tgamma(1. + 1./kappa);

    double logprior = - arma::accu(betas % betas) / tauSq / 2.;

    arma::vec logpost_first = std::log(kappa) -
                              kappa * (logMu - std::lgamma(1. + 1./kappa)) + 
                              (kappa - 1) * arma::log(dataclass.y);
    double logpost_first_sum = arma::sum( logpost_first.elem(arma::find(dataclass.event == 1)) );

    arma::vec logpost_second = arma::pow( dataclass.y / lambdas, kappa);
    logpost_second.elem(arma::find(logpost_second > upperbound9)).fill(upperbound9);
    double logpost_second_sum =  arma::sum( - logpost_second );

    return logpost_first_sum + logpost_second_sum + logprior;
}

