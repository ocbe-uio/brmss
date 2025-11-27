/* Log-likelihood for the use in Metropolis-Hastings sampler*/

#include "simple_gibbs.h"
#include "arms_gibbs.h"
#include "BVS_subfunc.h"
#include "BVS_logistic.h"

void BVS_logistic::mcmc(
    unsigned int nIter,
    unsigned int burnin,
    unsigned int thin,
    double& tau0Sq_,
    arma::vec& tauSq_,
    arma::mat& betas,
    arma::umat& gammas,
    const std::string& gammaProposal,
    Gamma_Sampler_Type gammaSampler,
    const std::string& rw_mh,
    const armsParmClass& armsPar,
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

    arma::vec loglik = loglikelihood(
                           betas,
                           dataclass
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
        sampleGamma(
            gammas,
            gammaSampler,
            gammaProposal,
            rw_mh,
            logP_gamma,
            gamma_acc_count,
            logP_beta,
            loglik,
            armsPar,
            hyperpar,
            betas,
            tau0Sq,
            tauSq,
            m,
            burnin,

            dataclass
        );

// std::cout << "...debug4\n";
        // update \betas
        betas.elem(arma::find(gammas == 0)).fill(0.);
        ARMS_Gibbs::arms_gibbs_beta_logistic(
            armsPar,
            hyperpar,
            betas,
            gammas,
            tau0Sq,
            tauSq,
            dataclass
        );

        // logP_beta = BVS_subfunc::logPDFNormal(betas, tauSq);

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

            loglik = loglikelihood(
                         betas,
                         dataclass
                     );

            // save loglikelihoods
            loglikelihood_mcmc.row(1+nIter_thin_count) = loglik.t();

            ++nIter_thin_count;
        }

    }

}

// individual loglikelihoods
arma::vec BVS_logistic::loglikelihood(
    const arma::mat& betas,
    const DataClass &dataclass)
{

    arma::vec xb = dataclass.X * betas;
    // xb.elem(arma::find(xb > 50.0)).fill(50.0);
    // xb.elem(arma::find(xb < -50.0)).fill(-50.0);

    arma::vec prob = 1.0 / (1.0 + arma::exp(-xb));
    prob.elem(arma::find(prob > 1.0-lowerbound0)).fill(1.0-lowerbound0);
    prob.elem(arma::find(prob < lowerbound0)).fill(lowerbound0);

    arma::vec loglik = dataclass.y % arma::log(prob) + (1.0-dataclass.y) % arma::log(1.0-prob) ;

    return loglik;
}


void BVS_logistic::sampleGamma(
    arma::umat& gammas,
    Gamma_Sampler_Type gamma_sampler,
    const std::string& gammaProposal,
    const std::string& rw_mh,
    arma::mat& logP_gamma,
    unsigned int& gamma_acc_count,
    double& logP_beta,
    arma::vec& loglik,
    const armsParmClass& armsPar,
    const hyperparClass& hyperpar,

    arma::mat& betas,
    double tau0Sq,
    double tauSq,
    const unsigned int iter,
    const unsigned int burnin,

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

    // adaptive factor for reaching MH acceptance rate 0.234 (Roberts GO and Rosenthal JS, 2001)
    // Note1: The target acceptance rate 0.234 can be difficult to reach due to very sparse prior on gamma
    // Note2: As Roberts & Rosenthal (2001) mentioned: there is very little to be gained from fine tuning of acceptance rates. ok with acc_rate (0.1, 0.4)
    static double a = std::log(2.38 * 2.38 / 4.0); // 'a' must be static variable

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
    double logPriorGammaRatio = 0.;
    proposedGammaPrior = logP_gamma; // copy the original one and later change the address of the copied one

    // TODO: check if pi0 is needed
    // double pi = pi0;
    for(auto i: updateIdx)
    {
        double pi = R::rbeta(hyperpar.piA + (double)(gammas(1+i,componentUpdateIdx)),
                             hyperpar.piB + 1 - (double)(gammas(1+i,componentUpdateIdx)));
        proposedGammaPrior(1+i,componentUpdateIdx) = BVS_subfunc::logPDFBernoulli( proposedGamma(1+i,componentUpdateIdx), pi );
        logPriorGammaRatio +=  proposedGammaPrior(1+i, componentUpdateIdx) - logP_gamma(1+i, componentUpdateIdx);
    }

    arma::mat proposedBeta = betas;

    // update (addresses) 'proposedBeta' and 'logPosteriorBeta_proposal' based on 'proposedGamma'
    proposedBeta.elem(arma::find(proposedGamma == 0)).fill(0.);

    // // Note that intercept is updated here
    /*
    ARMS_Gibbs::arms_gibbs_beta_logistic(
        armsPar,
        hyperpar,
        proposedBeta,
        proposedGamma,
        tauSq,
        tau0Sq,
        dataclass
    );
    */

    /////////////////////////////////////
    // random-walk proposal for betas
    /////////////////////////////////////
    /*
    unsigned int J = arma::sum(updateIdx) + 1; // plus intercept
    arma::mat Lambda = arma::zeros<arma::mat>(J, J);
    arma::vec diag_elements = arma::vec(J, arma::fill::value(1./tauSq));
    diag_elements[0] = 1./tau0Sq;
    Lambda.diag() = diag_elements; // inverse of prior variances

    arma::vec mu = 1. / (1. + arma::exp(betas[0] + dataclass.X * betas(1,0,p,0)));
    arma::mat W = arma::zeros<arma::mat>(N, N);
    // W.diag() = - dataclass.y / mu / mu + (1. - dataclass.y) / (1.-mu) / (1.-mu); // observed Hessian
    W.diag() = 1. / mu / (1. - mu); // Fisher weights, i.e. expected Hessian

    arma::mat G_J = 1. / W * arma::join_rows(arma::ones<arma::vec>(N) ,dataclass.X);

    // plus curvature H_J
    Lambda += G_J.t() * W * G_J;

    arma::mat invLambda;
    if( !arma::inv_sympd( invLambda,  Lambda ) )
    {
        arma::inv(invLambda, Lambda, arma::inv_opts::allow_approx);
    }
    SigmaRW = 2.38 * 2.38 / (double)J * invLambda;
    arma::vec beta_mask = BVS_subfunc::randMvNormal( arma::zeros<arma::vec>(J), SigmaRW );
    proposedBeta[0] = beta_mask[0];
    proposedBeta(updateIdx + 1) = beta_mask.subvec(1, J-1);
    */
    
    if (rw_mh != "symmetric") 
    {
        // step size of the random-walk proposal. Larger c increases proposal variance and typically lowers acceptance; smaller c increases acceptance but mixes slower.
        // double c = 2.38 * 2.38 / (double)J;
        double c = std::exp(a);
        if( arma::any(gammas(1+updateIdx)) )
        {
            /*
            // mu = 1. / (1. + arma::exp(-dataclass.X * betas));
            mu = 0.5 * (1. + arma::tanh((dataclass.X * betas) / 2.)); // more stable sigmoid
            mu.elem(arma::find(mu > 1.0-lowerbound0)).fill(1.0-lowerbound0);
            mu.elem(arma::find(mu < lowerbound0)).fill(lowerbound0);

            // W.diag() = - dataclass.y / mu / mu + (1. - dataclass.y) / (1.-mu) / (1.-mu); // observed Hessian
            W.diag() = mu % (1. - mu); // Fisher weights, i.e. expected Hessian
            // plus Gauss–Newton/Fisher block curvature H_J
            Lambda +=  dataclass.X.cols(1+updateIdx).t() * W * dataclass.X.cols(1+updateIdx);
            // std::cout << "...debug W.diag=" << W.diag().t() << "\n";
            // std::cout << "...debug Lambda=" << Lambda;
            Lambda.diag() += lambda0; //plus Damping 0.1 in diagonals
            if( !arma::inv_sympd( invLambda,  Lambda ) )
            {
                arma::inv(invLambda, Lambda, arma::inv_opts::allow_approx);
            }
            */
            arma::mat SigmaRW = c * calculateLambda(betas, tauSq, updateIdx, dataclass);
            arma::vec u = BVS_subfunc::randMvNormal( arma::zeros<arma::vec>(updateIdx.n_elem), SigmaRW );
            proposedBeta(1 + updateIdx) += u;
            proposedBeta.elem(arma::find(proposedGamma == 0)).fill(0.);

            // proposal: q(beta_proposal | beta); forward beta proposal density
            logProposalRatio -= BVS_subfunc::logPDFNormal(proposedBeta(1+updateIdx), betas(1+updateIdx), SigmaRW);
        }

        // proposal: q(beta | beta_proposal); reverse beta proposal density
        arma::mat SigmaRW_reverse = c * calculateLambda(proposedBeta, tauSq, updateIdx, dataclass);
        logProposalRatio += BVS_subfunc::logPDFNormal(betas(1+updateIdx), proposedBeta(1+updateIdx), SigmaRW_reverse);
    } else if( arma::any(gammas(1+updateIdx)) ) {
        // (symmetric) random-walk Metropolis with optimal standard deviation O(d^{-1/2})
        arma::vec u = Rcpp::rnorm( updateIdx.n_elem, 0., 1. / std::sqrt(updateIdx.n_elem) ); 
        proposedBeta(1 + updateIdx) += u;
        proposedBeta.elem(arma::find(proposedGamma == 0)).fill(0.); // assure some of proposed betas be 0 corresponding to proposed gammas 0
    }

    // prior ratio of beta
    double logPriorBetaRatio = 0.;
    if (gammaProposal == "simple")
        Rcpp::Rcout << "Warning: The argument 'gammaProposal = simple' is invalid!";

    logPriorBetaRatio = BVS_subfunc::logPDFNormal(proposedBeta(1+updateIdx), tauSq) - BVS_subfunc::logPDFNormal(betas(1+updateIdx), tauSq);

    // compute logLikelihoodRatio, i.e. proposedLikelihood - loglik
    arma::vec proposedLikelihood = loglikelihood( proposedBeta, dataclass );

    double logLikelihoodRatio = arma::sum(proposedLikelihood - loglik);

    // Here we need always compute the proposal and original ratios, in particular the likelihood, since betas are updated
    double logAccProb = logLikelihoodRatio +
                        logPriorGammaRatio +
                        logPriorBetaRatio +
                        logProposalRatio;

    bool accepted = (std::log(R::runif(0,1)) < logAccProb);
    if( accepted )
    {
        gammas = proposedGamma;
        logP_gamma = proposedGammaPrior;
        // logP_beta = proposedBetaPrior;
        loglik = proposedLikelihood;
        betas = proposedBeta;

        ++gamma_acc_count;
    }

    // Robbins–Monro update (during burn-in only) (Robbins H and Monro S, 1951)
    if (iter < burnin && rw_mh == "adaptive")
    {
        /*
        double acc_target = (J >= 5 ? 0.234 : 0.35);
        double a = std::log(2.38 * 2.38 / std::max(1u, J));
        double eta0 = 0.1;
        unsigned int m0 = 50;

        // Each MH proposal during burn-in:
        double c = std::exp(a);
        bool accepted = (std::log(R::runif(0,1)) < logAccProb);
        // Robbins–Monro update (during burn-in only)
        double eta_m = eta0 / (iter + m0);
        a += eta_m * ((accepted ? 1.0 : 0.0) - acc_target);
        */
        a += 0.1 / (iter + 50) * ((accepted ? 1.0 : 0.0) - 0.234);
        // std::cout << "...Adaptive factor for step size of the RW proposal a=" << a;
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

arma::mat BVS_logistic::calculateLambda(
    const arma::mat& betas,
    const double tauSq,
    const arma::uvec& updateIdx,
    const DataClass& dataclass)
{
        unsigned int J = updateIdx.n_elem; // not include intercept

        arma::vec mu = 0.5 * (1. + arma::tanh((dataclass.X * betas) / 2.)); // more stable sigmoid
        mu.elem(arma::find(mu > 1.0-lowerbound0)).fill(1.0-lowerbound0);
        mu.elem(arma::find(mu < lowerbound0)).fill(lowerbound0);

        // Hessian matrix
        // arma::mat W = arma::zeros<arma::mat>(N, N);
        // W.diag() = - dataclass.y / mu / mu + (1. - dataclass.y) / (1.-mu) / (1.-mu); // observed Hessian
        arma::mat W = arma::diagmat(mu % (1. - mu)); // Fisher weights, i.e. expected Hessian
        
        // add Gauss–Newton/Fisher block curvature H_J
        arma::mat Lambda = arma::zeros<arma::mat>(J, J);
        Lambda.diag() += 1./tauSq; // inverse of prior variances
        Lambda +=  dataclass.X.cols(1+updateIdx).t() * W * dataclass.X.cols(1+updateIdx);
        // lambda0 primarily as a numerical-stability parameter (ridge damping) and adjust it reactively when curvature/inversion issues arise
        Lambda.diag() += 0.1; //plus damping lambda0=0.1 in diagonals

        arma::mat invLambda;
        if( !arma::inv_sympd( invLambda,  Lambda ) )
        {
            arma::inv(invLambda, Lambda, arma::inv_opts::allow_approx);
        }

        return invLambda;
}

/*
double BVS_logistic::logPBeta(
    const arma::mat& betas,
    double tauSq,
    const DataClass& dataclass)
{
    double logP = 0.;
    logP = arma::sum(loglikelihood( betas, dataclass)) - 0.5 * arma::accu(betas % betas)/tauSq;

    return logP;
}
*/
