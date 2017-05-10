#include "header.h"
#include <math.h>

void error_norm(double rms[5])
{
    int i, j, k, m;
    double xi, eta, zeta, u_exact[5], add;

    for (m = 0; m < 5; m++)
        rms[m] = 0.0;

    for (k = 0; k <= nz - 1; k++) 
    {
        for (j = 0; j <= ny - 1; j++) 
        {
            for (i = 0; i <= nx - 1; i++) 
            {
                zeta = k * dnzm1;
                eta = j * dnym1;
                xi = i * dnxm1;
                
                exact_solution(xi, eta, zeta, u_exact);
                
                for (m = 0; m < 5; m++) 
                {
                    add = u(k,j,i,m) - u_exact[m];
                    rms[m] = rms[m] + add*add;
                }
            }
        }
    }

    for (m = 0; m < 5; m++) 
    {
        rms[m] = rms[m] / nx2;
        rms[m] = rms[m] / ny2;
        rms[m] = rms[m] / nz2;

        rms[m] = sqrt(rms[m]);
    }
}


void rhs_norm(double rms[5])
{
    int i, j, k, m;
    double add;

    for (m = 0; m < 5; m++)
        rms[m] = 0.0;

    for (k = 1; k <= nz2; k++) 
    {
        for (j = 1; j <= ny2; j++) 
        {
            for (i = 1; i <= nx2; i++) 
            {
                for (m = 0; m < 5; m++) 
                {
                    add = rhs(k,j,i,m);
                    rms[m] = rms[m] + add*add;
                }
            }
        }
    }

    for (m = 0; m < 5; m++) 
    {
        rms[m] = rms[m] / nx2;
        rms[m] = rms[m] / ny2;
        rms[m] = rms[m] / nz2;

        rms[m] = sqrt(rms[m]);
    }
}
