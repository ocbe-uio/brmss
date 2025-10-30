/* subfunctions for BVS_*/

#include "BVS_subfunc.h"

double BVS_subfunc::gammaMC3Proposal(
    unsigned int p,
    arma::umat& mutantGamma,
    const arma::umat gammas,
    arma::uvec& updateIdx,
    unsigned int componentUpdateIdx )
{
    //arma::umat mutantGamma = gammas;
    unsigned int n_updates_MC3 = std::max(5., std::ceil( (double)(p) / 5. )); //arbitrary number

    Rcpp::IntegerVector entireIdx = Rcpp::seq(0, p - 1);
    updateIdx = Rcpp::as<arma::uvec>(Rcpp::sample(entireIdx, n_updates_MC3, false)); // here 'replace = false'

    for( auto i : updateIdx)
    {
        mutantGamma(1+i,componentUpdateIdx) = ( R::runif(0,1) < 0.5 )? gammas(1+i,componentUpdateIdx) : 1-gammas(1+i,componentUpdateIdx); // could simply be ( 0.5 ? 1 : 0) ;
    }

    //return mutantGamma ;
    return 0. ; // pass this to the outside, it's the (symmetric) logProposalRatio
}

// sampler for proposed updates on gammas
double BVS_subfunc::gammaBanditProposal(
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
        mismatch(j) = (mutantGamma(1+j,componentUpdateIdx)==0)?(banditZeta(j)):(1.-banditZeta(j));   //mismatch
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
        mutantGamma(1+updateIdx(0),componentUpdateIdx) = 1 - gammas(1+updateIdx(0),componentUpdateIdx); // deterministic, just switch

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
            mutantGamma(1+updateIdx(i),componentUpdateIdx) = j;

            normalised_mismatch_backwards(updateIdx(i)) = 1.- normalised_mismatch_backwards(updateIdx(i));

            logProposalRatio += logPDFBernoulli(gammas(1+updateIdx(i),componentUpdateIdx),banditZeta(updateIdx(i))) -
                                logPDFBernoulli(mutantGamma(1+updateIdx(i),componentUpdateIdx),banditZeta(updateIdx(i)));
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

arma::uvec BVS_subfunc::randWeightedIndexSampleWithoutReplacement(
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
unsigned int BVS_subfunc::randWeightedIndexSampleWithoutReplacement(
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
double BVS_subfunc::logPDFWeightedIndexSampleWithoutReplacement(
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

double BVS_subfunc::logspace_add(
    double a,
    double b)
{

    if(a <= std::numeric_limits<float>::lowest())
        return b;
    if(b <= std::numeric_limits<float>::lowest())
        return a;
    return std::max(a, b) + std::log( (double)(1. + std::exp( (double)-std::abs((double)(a - b)) )));
}

double BVS_subfunc::logPDFBernoulli(unsigned int x, double pi)
{
    if( x > 1 ||  x < 0 )
        return -std::numeric_limits<double>::infinity();
    else
        return (double)(x) * std::log(pi) + (1.-(double)(x)) * std::log(1. - pi);
}

double BVS_subfunc::logPDFNormal(double x, double sigmaSq)  // zeroMean
{

    return -0.5*log(2.*M_PI) -0.5*std::log(sigmaSq) - 0.5 * x * x / sigmaSq;

}

double BVS_subfunc::logPDFNormal(const arma::vec& x, double sigmaSq)  // zeroMean and independentVar
{
    unsigned int k = x.n_elem;
    double tmp = (double)k * std::log(sigmaSq); // log-determinant(Sigma)

    return -0.5*(double)k*log(2.*M_PI) -0.5*tmp - 0.5 * arma::as_scalar( x.t() * x ) / sigmaSq;

}
