#include <stdio.h>
#include "header.h"

void print_results(int niter, double tmax, logical verified, const char **t_names)
{ 
    double mflops;
    double trecs[t_last + 1];
    int i;

    if (tmax != 0.0)
    {
        int n3 = nx * ny * nz;
        double t = (nx + ny + nz) / 3.0;
        mflops = (881.174 * (double)n3 - 4683.91 * (t * t) + 11484.5 * t - 19272.4) * (double)niter / (tmax * 1000000.0);
    }
    else
        mflops = 0.0;

    printf("\n\n Program completed.\n");
    printf(" Size            =           %4dx%4dx%4d\n", nx, ny, nz);
    printf(" Iterations      =             %12d\n", niter);
    printf(" Time in seconds =             %12.2lf\n", tmax);
    printf(" Mop/s total     =          %15.2lf\n", mflops);
    if (verified)
        printf(" Verification    =             %12s\n", "SUCCESSFUL");
    else
        printf(" Verification    =             %12s\n", "UNSUCCESSFUL");

    if (timeron)
    {
        for (i = 1; i <= t_last; i++)
            trecs[i] = timer_read(i);

        if (tmax == 0.0) 
            tmax = 1.0;

        printf("\n  SECTION       Time (seconds)\n");
        printf(" ================================\n");
        printf("  %-8s:%9.3f  (%6.2f%%)\n", t_names[t_total], trecs[t_total], trecs[t_total] * 100. / tmax);
        printf(" ================================\n");

        printf("  %-8s:%9.3f  (%6.2f%%)\n", t_names[t_rhs], trecs[t_rhs], trecs[t_rhs] * 100. / tmax);
        printf("    --> %-8s %9.3f  (%6.2f%%)\n", t_names[t_rhsx], trecs[t_rhsx], trecs[t_rhsx] * 100. / tmax);
        printf("    --> %-8s %9.3f  (%6.2f%%)\n", t_names[t_rhsy], trecs[t_rhsy], trecs[t_rhsy] * 100. / tmax);
        printf("    --> %-8s %9.3f  (%6.2f%%)\n", t_names[t_rhsz], trecs[t_rhsz], trecs[t_rhsz] * 100. / tmax);

        printf("  %-8s:%9.3f  (%6.2f%%)\n", t_names[t_txinvr], trecs[t_txinvr], trecs[t_txinvr] * 100. / tmax);

        printf("  %-8s:%9.3f  (%6.2f%%)\n", t_names[t_xsolve], trecs[t_xsolve], trecs[t_xsolve] * 100. / tmax);
        //printf("    --> %-8s %9.3f  (%6.2f%%)\n", t_names[t_ninvr], trecs[t_ninvr], trecs[t_ninvr] * 100. / tmax);
        printf("  %-8s:%9.3f  (%6.2f%%)\n", t_names[t_ysolve], trecs[t_ysolve], trecs[t_ysolve] * 100. / tmax);
        //printf("    --> %-8s %9.3f  (%6.2f%%)\n", t_names[t_pinvr], trecs[t_pinvr], trecs[t_pinvr] * 100. / tmax);
        printf("  %-8s:%9.3f  (%6.2f%%)\n", t_names[t_zsolve], trecs[t_zsolve], trecs[t_zsolve] * 100. / tmax);
        //printf("    --> %-8s %9.3f  (%6.2f%%)\n", t_names[t_tzetar], trecs[t_tzetar], trecs[t_tzetar] * 100. / tmax);

        printf("  %-8s:%9.3f  (%6.2f%%)\n", t_names[t_add], trecs[t_add], trecs[t_add] * 100. / tmax);
        
    }
}
