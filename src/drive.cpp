// Main function for the MCMC loop


#include "dirichlet.h"
#include "weibull.h"

#ifdef _OPENMP
extern omp_lock_t RNGlock; /*defined in global.h*/
#include <omp.h>
#endif

#include <Rcpp.h>
// [[Rcpp::plugins(openmp)]]

#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]


//' Main function implemented in C++ for the MCMC loop
//'
//' @name run_mcmc
//'
//' @param y response variable. A vector, matrix or dataframe
//' @param X input matrix or dataframe
//' @param L number of response variables
//' @param family a character string representing one of the built-in families
//' @param nIter the number of iterations of the chain
//' @param burnin number of iterations to discard at the start of the chain
//' @param thin thinning MCMC intermediate results to be stored
//' @param tick an integer used for printing the iteration index and some updated
//' parameters every tick-th iteration. Default is 1
//' @param gamma_sampler M-H sampler with "mc" or multi-armed "bandit" proposal for gammas
//' @param gamma_proposal one of 'c("simple", "posterior")'
//' @param threads number of threads used for parallelization. Default is 1
//' @param n number of samples to draw
//' @param nsamp how many samples to draw for generating each sample; only the last draw will be kept
//' @param ninit number of initials as meshgrid values for envelop search
//' @param convex adjustment for convexity (non-negative value, default 1.0)
//' @param npoint maximum number of envelope points
//' @param initList a list of initial values for parameters "kappa", "betas"
//' @param rangeList a list of ranges of initial values for parameters "kappa", "betas"
//' @param hyperparList a list of relevant hyperparameters
//'
// [[Rcpp::export]]
Rcpp::List run_mcmc(
    arma::mat& y,
    arma::mat& X,
    unsigned int L,

    const std::string& family,
    unsigned int nIter,
    unsigned int burnin,
    unsigned int thin,
    unsigned int tick,
    const std::string& gamma_sampler,
    const std::string& gamma_proposal,
    int threads,

    unsigned int n,
    int nsamp,
    int ninit,
    double convex,
    int npoint,

    const Rcpp::List& initList,
    const Rcpp::List& rangeList,
    const Rcpp::List& hyperparList)
{
#ifdef _OPENMP
    // omp_set_nested( 0 );
    // omp_set_num_threads( 1 );
    if( threads == 1 )
    {
        omp_set_nested( 0 );
        omp_set_num_threads( 1 );
    }
    else
    {
        omp_init_lock(&RNGlock);  // init RNG lock for the parallel part

        omp_set_nested(0); // 1=enable, 0=disable nested parallelism (e.g. compute likelihoods in parallel at least wrt to outcomes + wrt to individuals)
        omp_set_num_threads( threads ); // TODO: 'threads' seems not faster always
    }
#endif

    // dimensions
    unsigned int N = X.n_rows;
    unsigned int p = X.n_cols;
    // unsigned int L = y.n_cols;

    // arms parameters in a class
    int metropolis = 1;
    armsParmClass armsPar(n, nsamp, ninit, metropolis, convex, npoint,
                          Rcpp::as<double>(rangeList["kappaMin"]),
                          Rcpp::as<double>(rangeList["kappaMax"]),
                          Rcpp::as<double>(rangeList["betaMin"]),
                          Rcpp::as<double>(rangeList["betaMax"]));


    // hyperparameters
    double tauSq = 1.;
    double tau0Sq = 1.;

    hyperparClass hyperpar(
        Rcpp::as<double>(hyperparList["piA"]),
        Rcpp::as<double>(hyperparList["piB"]),
        Rcpp::as<double>(hyperparList["tau0A"]),
        Rcpp::as<double>(hyperparList["tau0B"]),
        Rcpp::as<double>(hyperparList["tauA"]),
        Rcpp::as<double>(hyperparList["tauB"]),
        Rcpp::as<double>(hyperparList["kappaA"]),
        Rcpp::as<double>(hyperparList["kappaB"])
    );

    // hyperparList = Rcpp::List();  // Clear it by creating a new empty List

    // response family
    Family_Type familyType;
    if ( family == "weibull" )
        familyType = Family_Type::weibull;
    else if ( family == "dirichlet" )
        familyType = Family_Type::dirichlet ;
    else
    {
        Rprintf("ERROR: Wrong type of family given!");
        return 1;
    }
    // TODO: add more families

    // Gamma Sampler
    Gamma_Sampler_Type gammaSampler;
    if ( gamma_sampler == "bandit" )
        gammaSampler = Gamma_Sampler_Type::bandit;
    else if ( gamma_sampler == "mc3" )
        gammaSampler = Gamma_Sampler_Type::mc3 ;
    else
    {
        Rprintf("ERROR: Wrong type of Gamma Sampler given!");
        return 1;
    }

    // initial values of key parameters and save them in a struct object
    arma::mat betas = Rcpp::as<arma::mat>(initList["betas"]);
    double kappa = Rcpp::as<double>(initList["kappa"]);
    // initList = Rcpp::List();  // Clear it by creating a new empty List

    unsigned int nIter_thin = nIter / thin;
    // initializing mcmc results
    arma::vec tauSq_mcmc = arma::zeros<arma::vec>(1+nIter_thin);
    tauSq_mcmc[0] = tauSq; // TODO: only keep the first one for now
    arma::mat beta_mcmc = arma::zeros<arma::mat>(1+nIter_thin, (p+1)*L);
    beta_mcmc.row(0) = arma::vectorise(betas).t();
    arma::vec kappa_mcmc = arma::zeros<arma::vec>(1+nIter_thin);
    kappa_mcmc[0] = kappa;

    // initializing relevant quantities; can be declared like arma::mat&

    // quantity 01
    arma::umat gammas = arma::ones<arma::umat>(p, L);
    unsigned int gamma_acc_count; // count acceptance of gammas via M-H sampler
    arma::umat gamma_mcmc = arma::zeros<arma::umat>(1+nIter_thin, p*L);
    gamma_mcmc.row(0) = arma::vectorise(gammas).t();

    // mean parameter
    arma::mat mu = arma::zeros<arma::mat>(N, L);
    // arma::vec mu;
    arma::vec logMu = arma::zeros<arma::mat>(N, L);

    arma::uvec event;
    arma::vec lambdas; // Weibull's scale parameter
    if(family == "weibull")
    {
        event = arma::conv_to<arma::uvec>::from( y.col(1) );
        y.shed_col(1);

        // arma::vec logMu = arma::join_cols(arma::ones<arma::rowvec>(N), X) * betas.col(0);
        logMu = betas(0) + X * betas.submat(1, 0, p, 0);
    }

    // input constant data sets in a class
    DataClass dataclass(X, y, event);
    X.clear();
    y.clear();
    event.clear();

    // initializing posterior mean
    double kappa_post = 0.;
    arma::mat beta_post = arma::zeros<arma::mat>(arma::size(betas));
    arma::umat gamma_post = arma::zeros<arma::umat>(arma::size(gammas));

    arma::mat loglikelihood_mcmc = arma::zeros<arma::mat>(1+nIter_thin, N);


    // ###########################################################
    // ## models for MCMC
    // ###########################################################

    switch( familyType )
    {
    case Family_Type::weibull:
        weibull(
            nIter,
            burnin,
            thin,
            kappa,
            tau0Sq,
            tauSq,
            betas,
            gammas,
            gamma_proposal,
            gammaSampler,
            familyType,
            armsPar,
            hyperpar,
            dataclass,

            kappa_mcmc,
            kappa_post,
            beta_mcmc,
            beta_post,
            gamma_mcmc,
            gamma_post,
            gamma_acc_count,
            loglikelihood_mcmc,
            tauSq_mcmc
        );
        break;

    case Family_Type::dirichlet:
        dirichlet(
            nIter,
            burnin,
            thin,
            tau0Sq,
            tauSq,
            betas,
            gammas,
            gamma_proposal,
            gammaSampler,
            familyType,
            armsPar,
            hyperpar,
            dataclass,

            beta_mcmc,
            beta_post,
            gamma_mcmc,
            gamma_post,
            gamma_acc_count,
            loglikelihood_mcmc,
            tauSq_mcmc
        );
        break;
    }


    Rcpp::Rcout << "\n";

    // wrap all outputs
    Rcpp::List output_mcmc;
    output_mcmc["kappa"] = kappa_mcmc;
    //output["phi"] = phi_mcmc;
    output_mcmc["betas"] = beta_mcmc;
    output_mcmc["gammas"] = gamma_mcmc;
    output_mcmc["gamma_acc_rate"] = ((double)gamma_acc_count) / ((double)nIter);
    // arma::mat gamma_post_mean = arma::zeros<arma::mat>(arma::size(gamma_post));
    arma::mat gamma_post_mean = arma::conv_to<arma::mat>::from(gamma_post) / ((double)(nIter - burnin));

    output_mcmc["loglikelihood"] = loglikelihood_mcmc;
    output_mcmc["tauSq"] = tauSq_mcmc;

    kappa_post /= ((double)(nIter - burnin));
    beta_post /= ((double)(nIter - burnin));
    output_mcmc["post"] = Rcpp::List::create(
                              Rcpp::Named("kappa") = kappa_post,
                              Rcpp::Named("betas") = beta_post,
                              Rcpp::Named("gammas") = gamma_post_mean
                          );

    return output_mcmc;
}

