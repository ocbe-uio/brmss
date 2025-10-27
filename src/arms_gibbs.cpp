// Gibbs sampling for univariate and multivariate ARMS

#include "arms_gibbs.h"
#include "simple_gibbs.h"

// Multivariate ARMS via Gibbs sampler for all responses of Dirichlet model
void ARMS_Gibbs::arms_gibbs_beta_dirichlet(
    const armsParmClass& armsPar,
    const hyperparClass& hyperpar,
    arma::mat& currentPars,
    arma::umat gammas,
    arma::vec& tauSq,
    double& tau0Sq,
    const DataClass &dataclass
)
{
    // dimensions
    unsigned int N = dataclass.X.n_rows;
    unsigned int p = dataclass.X.n_cols;
    unsigned int L = dataclass.y.n_cols;

    // logPosteriorBeta = 0.; // reset value 0

    // objects for arms()
    double minD = armsPar.betaMin;
    double maxD = armsPar.betaMax;

    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA

    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];

    dataS *mydata = (dataS *)malloc(sizeof (dataS));

    gammas = arma::join_cols(arma::ones<arma::urowvec>(L), gammas);
    currentPars.elem(arma::find(gammas == 0)).fill(0.);
    mydata->currentPars = currentPars.memptr();
    mydata->N = N;
    mydata->p = p;
    mydata->L = L;
    mydata->X = dataclass.X.memptr();
    mydata->y = dataclass.y.memptr();

    for (unsigned int l = 0; l < L; ++l)
    {
        // Gibbs sampling
        tau0Sq = sampleTau(hyperpar.tau0A, hyperpar.tau0B, currentPars.row(0).t()); // TODO: it seems not better if fixing tau0Sq=10
        mydata->tau0Sq = tau0Sq;
        tauSq[l] = sampleTau(hyperpar.tauA, hyperpar.tauB, currentPars.submat(1,l,p,l));
        mydata->tauSq = tauSq[l];

        for (unsigned int j = 0; j < p+1; ++j)
        {
            if (gammas(j, l))
            {
                mydata->jj = j;
                mydata->l = l;

                double xprev = currentPars(j);
                std::vector<double> xsamp(armsPar.nsamp);

                double qcent[1], xcent[1];
                int neval, ncent = 0;

                int err;
                double convex = armsPar.convex;
                err = ARMS::arms (
                          xinit.data(), armsPar.ninit, &minD, &maxD,
                          EvalFunction::log_dens_betas_dirichlet, mydata,
                          &convex, armsPar.npoint,
                          armsPar.metropolis, &xprev, xsamp.data(),
                          armsPar.nsamp, qcent, xcent, ncent, &neval);

                // check ARMS validity
                if (err > 0)
                    Rprintf("In arms_gibbs_beta(): error code in ARMS = %d.\n", err);
                if (std::isnan(xsamp[armsPar.nsamp-1]))
                    Rprintf("In arms_gibbs_beta(): NaN generated, possibly due to overflow in (log-)density (e.g. with densities involving exp(exp(...))).\n");
                if (xsamp[armsPar.nsamp-1] < minD || xsamp[armsPar.nsamp-1] > maxD)
                    Rprintf("In arms_gibbs_beta(): %d-th sample out of range [%f, %f] (fused domain). Got %f.\n", armsPar.nsamp, minD, maxD, xsamp[armsPar.nsamp-1]);

                currentPars(j, l) = xsamp[armsPar.nsamp - 1];

            }
        }

    }

    free(mydata);
}

// Multivariate ARMS via Gibbs sampler for one response of Dirichlet model
void ARMS_Gibbs::arms_gibbs_betaK_dirichlet(
    const unsigned int k,
    const armsParmClass& armsPar,
    const hyperparClass& hyperpar,
    arma::mat& currentPars,
    arma::umat gammas,
    double tauSqK,
    double tau0Sq,
    const DataClass &dataclass
)
{
    // dimensions
    unsigned int N = dataclass.X.n_rows;
    unsigned int p = dataclass.X.n_cols;
    unsigned int L = dataclass.y.n_cols;

    // logPosteriorBeta = 0.; // reset value 0

    // objects for arms()
    double minD = armsPar.betaMin;
    double maxD = armsPar.betaMax;

    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA

    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];

    dataS *mydata = (dataS *)malloc(sizeof (dataS));

    gammas = arma::join_cols(arma::ones<arma::urowvec>(L), gammas);
    currentPars.elem(arma::find(gammas == 0)).fill(0.);
    mydata->currentPars = currentPars.memptr();
    mydata->N = N;
    mydata->p = p;
    mydata->L = L;
    mydata->tau0Sq = tau0Sq;
    mydata->tauSq = tauSqK;
    mydata->X = dataclass.X.memptr();
    mydata->y = dataclass.y.memptr();

    // Gibbs sampling
    unsigned int l = k;
    mydata->l = l;

    for (unsigned int j = 0; j < p+1; ++j)
    {
        if (gammas(j, l))
        {
            mydata->jj = j;

            double xprev = currentPars(j);
            std::vector<double> xsamp(armsPar.nsamp);

            double qcent[1], xcent[1];
            int neval, ncent = 0;

            int err;
            double convex = armsPar.convex;
            err = ARMS::arms (
                      xinit.data(), armsPar.ninit, &minD, &maxD,
                      EvalFunction::log_dens_betas_dirichlet, mydata,
                      &convex, armsPar.npoint,
                      armsPar.metropolis, &xprev, xsamp.data(),
                      armsPar.nsamp, qcent, xcent, ncent, &neval);

            // check ARMS validity
            if (err > 0)
                Rprintf("In arms_gibbs_beta(): error code in ARMS = %d.\n", err);
            if (std::isnan(xsamp[armsPar.nsamp-1]))
                Rprintf("In arms_gibbs_beta(): NaN generated, possibly due to overflow in (log-)density (e.g. with densities involving exp(exp(...))).\n");
            if (xsamp[armsPar.nsamp-1] < minD || xsamp[armsPar.nsamp-1] > maxD)
                Rprintf("In arms_gibbs_beta(): %d-th sample out of range [%f, %f] (fused domain). Got %f.\n", armsPar.nsamp, minD, maxD, xsamp[armsPar.nsamp-1]);

            currentPars(j, l) = xsamp[armsPar.nsamp - 1];

        }
    }

    free(mydata);
}

// Multivariate ARMS via Gibbs sampler for Weibull model
void ARMS_Gibbs::arms_gibbs_beta_weibull(
    const armsParmClass& armsPar,
    const hyperparClass& hyperpar,
    arma::mat& currentPars,
    arma::umat gammas,
    double& tauSq,
    double& tau0Sq,

    double kappa,
    const DataClass &dataclass
)
{
    // dimensions
    unsigned int N = dataclass.X.n_rows;
    unsigned int p = dataclass.X.n_cols;
    // unsigned int L = dataclass.y.n_cols;

    // logPosteriorBeta = 0.; // reset value 0

    // objects for arms()
    double minD = armsPar.betaMin;
    double maxD = armsPar.betaMax;

    std::vector<double> xinit(armsPar.ninit); // Use std::vector instead of VLA

    arma::vec xinit0 = arma::linspace( minD+1.0e-10, maxD-1.0e-10, armsPar.ninit );
    for (unsigned int i = 0; i < armsPar.ninit; ++i)
        xinit[i] = xinit0[i];

    dataS *mydata = (dataS *)malloc(sizeof (dataS));

    gammas = arma::join_cols(arma::ones<arma::urowvec>(1), gammas);
    currentPars.elem(arma::find(gammas == 0)).fill(0.);
    mydata->currentPars = currentPars.memptr();
    mydata->p = p;
    mydata->N = N;
    mydata->kappa = kappa;
    // mydata->mu = mu.memptr();
    mydata->X = dataclass.X.memptr();
    mydata->y = dataclass.y.memptr();
    mydata->event = dataclass.event.memptr();

    // Gibbs sampling

    tau0Sq = sampleTau(hyperpar.tau0A, hyperpar.tau0B, currentPars.row(0).t());
    mydata->tau0Sq = tau0Sq;
    tauSq = sampleTau(hyperpar.tauA, hyperpar.tauB, currentPars.submat(1,0,p,0));
    mydata->tauSq = tauSq;

    for (unsigned int j = 0; j < p+1; ++j)
    {
        if (gammas(j))
        {
            mydata->jj = j;

            double xprev = currentPars(j);
            std::vector<double> xsamp(armsPar.nsamp);

            double qcent[1], xcent[1];
            int neval, ncent = 0;

            int err;
            double convex = armsPar.convex;
            err = ARMS::arms (
                      xinit.data(), armsPar.ninit, &minD, &maxD,
                      EvalFunction::log_dens_betas_weibull, mydata,
                      &convex, armsPar.npoint,
                      armsPar.metropolis, &xprev, xsamp.data(),
                      armsPar.nsamp, qcent, xcent, ncent, &neval);

            // check ARMS validity
            if (err > 0)
                Rprintf("In arms_gibbs_beta(): error code in ARMS = %d.\n", err);
            if (std::isnan(xsamp[armsPar.nsamp-1]))
                Rprintf("In arms_gibbs_beta(): NaN generated, possibly due to overflow in (log-)density (e.g. with densities involving exp(exp(...))).\n");
            if (xsamp[armsPar.nsamp-1] < minD || xsamp[armsPar.nsamp-1] > maxD)
                Rprintf("In arms_gibbs_beta(): %d-th sample out of range [%f, %f] (fused domain). Got %f.\n", armsPar.nsamp, minD, maxD, xsamp[armsPar.nsamp-1]);

            currentPars(j) = xsamp[armsPar.nsamp - 1];

        }
    }

    // logPosteriorBeta = logPbetaK(k, currentPars, mydata->tauSq, kappa, datTheta, datProportion, dataclass);

    free(mydata);
}

//' Slice sampling for kappa
//'
//' @param n Number of samples to draw
//' @param nsamp How many samples to draw for generating each sample; only the last draw will be kept
//' @param ninit Number of initials as meshgrid values for envelop search
//' @param convex Adjustment for convexity (non-negative value, default 1.0)
//' @param npoint Maximum number of envelope points
//' @param dirichlet Not yet implemented
//'
void ARMS_Gibbs::slice_kappa(
    double& currentPars,
    double minD,
    double maxD,
    double kappaA,
    double kappaB,
    const DataClass &dataclass,
    arma::vec& logMu)
{
    // dimensions
    unsigned int N = dataclass.y.n_rows;

    dataS *mydata = (dataS *)malloc(sizeof (dataS));
    mydata->N = N;
    mydata->kappaA = kappaA;
    mydata->kappaB = kappaB;
    mydata->logMu = logMu.memptr();
    mydata->y = dataclass.y.memptr();
    mydata->event = dataclass.event.memptr();

    slice_sample (
        EvalFunction::log_dens_kappa,
        mydata,
        currentPars,
        10,
        1.0,
        minD,
        maxD
    );

    free(mydata);

}

void ARMS_Gibbs::slice_sample(
    double (*logfn)(double par, void *mydata),
    void *mydata,
    double& x,
    const unsigned int steps,
    const double w,
    const double lower,
    const double upper)
{
    double L_bound = 0.;
    double R_bound = 0.;
    double logy = logfn(x, mydata);

    // we can add omp parallelisation here
    for (unsigned int i = 0; i < steps; ++i)
    {
        // draw uniformly from [0, y]
        double logz = logy - R::rexp(1);

        // expand search range
        double u = R::runif(0.0, 1.0) * w;
        L_bound = x - u;
        R_bound = x + (w - u);
        while ( L_bound > lower && logfn(L_bound, mydata) > logz )
        {
            L_bound -= w;
        }
        while ( R_bound < upper && logfn(R_bound, mydata) > logz )
        {
            R_bound += w;
        }

        // sample until draw is within valid range
        double r0 = std::max(L_bound, lower);
        double r1 = std::min(R_bound, upper);

        double xs = x;
        double logys = 0.;
        int cnt = 0;
        do
        {
            cnt++;
            xs = R::runif(r0, r1);
            logys = logfn(xs, mydata);
            if ( logys > logz )
                break;
            if ( xs < x )
            {
                r0 = xs;
            }
            else
            {
                r1 = xs;
            }
        }
        while (cnt < 1e4);

        if (cnt == 1e4) ::Rf_error("slice_sample_cpp loop did not finish");

        x = xs;
        logy = logys;
    }

}