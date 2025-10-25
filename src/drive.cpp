// Main function for the MCMC loop


#include "simple_gibbs.h"
#include "arms_gibbs.h"
#include "BVS.h"
#include "global.h"

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
    arma::umat gamma_mcmc;
    arma::mat logP_gamma; // this is declared globally to be updated in the M-H sampler for gammas
    unsigned int gamma_acc_count; // count acceptance of gammas via M-H sampler
    logP_gamma = arma::zeros<arma::mat>(p, L);
    // }
    gamma_acc_count = 0;
    for(unsigned int l=0; l<L; ++l)
    {
        double pi = R::rbeta(hyperpar.piA, hyperpar.piB);

        for(unsigned int j=0; j<p; ++j)
        {
            gammas(j, l) = R::rbinom(1, pi);
            logP_gamma(j, l) = BVS_Sampler::logPDFBernoulli(gammas(j, l), pi);
        }
    }
    gamma_mcmc = arma::zeros<arma::umat>(1+nIter_thin, p*L);
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
        // logMu.elem(arma::find(logMu > upperbound)).fill(upperbound);
        // mu.col(0) = arma::exp( logMu );
        // lambdas = arma::pow( y / (mu.col(0) / std::tgamma(1. + 1./kappa)), kappa);
        // lambdas.elem(arma::find(lambdas > upperbound)).fill(upperbound);

        // // input constant data sets in a class
        // DataClass dataclass(event, y, X, N, p, L);
    }

    // input constant data sets in a class
    DataClass dataclass(X, y, event);
    X.clear();
    y.clear();
    event.clear();

    // quantity 3
    arma::mat proportion;
    if(family == "dirichlet")
    {
        arma::mat alphas = arma::zeros<arma::mat>(N, L);
        for(unsigned int l=0; l<L; ++l)
        {
            alphas.col(l) = arma::exp( betas(0, l) + dataclass.X * betas.submat(1, l, p, l) );
        }
        alphas.elem(arma::find(alphas > upperbound3)).fill(upperbound3);
        alphas.elem(arma::find(alphas < lowerbound)).fill(lowerbound);
        proportion = alphas / arma::repmat(arma::sum(alphas, 1), 1, L);
    }

    // initializing posterior mean
    double kappa_post = 0.;
    arma::mat beta_post = arma::zeros<arma::mat>(arma::size(betas));
    arma::umat gamma_post = arma::zeros<arma::umat>(arma::size(gammas));

    arma::mat loglikelihood_mcmc = arma::zeros<arma::mat>(1+nIter_thin, N);
    arma::vec loglik = arma::zeros<arma::vec>(N);
    BVS_Sampler::loglikelihood(
        betas,
        kappa,
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


        // update Weibull's shape parameter kappa
        ARMS_Gibbs::slice_kappa(
            kappa,
            armsPar.kappaMin,
            armsPar.kappaMax,
            hyperpar.kappaA,
            hyperpar.kappaB,
            dataclass,
            logMu
        );

        // update Weibull's quantities based on the new kappa
        // lambdas = mu.col(0) / std::tgamma(1.0+1.0/kappa);
        // weibullS = arma::exp(- arma::pow( y/lambdas, kappa));

        // update \gammas -- variable selection indicators
        BVS_Sampler::sampleGamma(
            gammas,
            gammaSampler,
            familyType,
            logP_gamma,
            gamma_acc_count,
            loglik,
            armsPar,
            hyperpar,
            betas,
            kappa,
            tau0Sq,
            tauSq,

            dataclass
        );

        // update \betas
        ARMS_Gibbs::arms_gibbs_beta_weibull(
            armsPar,
            hyperpar,
            betas,
            gammas,
            tauSq,
            tau0Sq,

            kappa,
            dataclass
        );

        // update \betas' variance tauSq
        // hyperpar.tauSq = sampleTau(hyperpar.tauA, hyperpar.tauB, betas);
        // tauSq_mcmc[1+m] = hyperpar.tauSq;

#ifdef _OPENMP
        #pragma omp parallel for
#endif

        // update Weibull's quantities based on the new betas
        for(unsigned int l=0; l<L; ++l)
        {
            logMu = betas(0) + dataclass.X * betas.submat(1, l, p, l);
            // logMu.elem(arma::find(logMu > upperbound)).fill(upperbound);
            // mu.col(l) = arma::exp( logMu );
        }

        // save results for un-thinned posterior mean
        if(m >= burnin)
        {
            kappa_post += kappa;
            beta_post += betas;
            gamma_post += gammas;
        }

        // save results of thinned iterations
        if((m+1) % thin == 0)
        {
            kappa_mcmc[1+nIter_thin_count] = kappa;
            beta_mcmc.row(1+nIter_thin_count) = arma::vectorise(betas).t();
            tauSq_mcmc[1+nIter_thin_count] = tauSq;//hyperpar.tauSq; // TODO: only keep the firs one for now

            gamma_mcmc.row(1+nIter_thin_count) = arma::vectorise(gammas).t();

            BVS_Sampler::loglikelihood(
                betas,
                kappa,
                dataclass,
                loglik
            );

            // save loglikelihoods
            loglikelihood_mcmc.row(1+nIter_thin_count) = loglik.t();

            ++nIter_thin_count;
        }

    }

    Rcpp::Rcout << "\n";

    // wrap all outputs
    Rcpp::List output_mcmc;
    output_mcmc["kappa"] = kappa_mcmc;
    //output["phi"] = phi_mcmc;
    output_mcmc["betas"] = beta_mcmc;
    arma::mat gamma_post_mean = arma::zeros<arma::mat>(arma::size(gamma_post));
    output_mcmc["gammas"] = gamma_mcmc;
    output_mcmc["gamma_acc_rate"] = ((double)gamma_acc_count) / ((double)nIter);
    gamma_post_mean = arma::conv_to<arma::mat>::from(gamma_post) / ((double)(nIter - burnin));

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

