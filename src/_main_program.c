//---------------------------------------------------------------------
// MAIN program 
//---------------------------------------------------------------------

#include <stdio.h>
#include <stdlib.h>
#include "header.h"

/* global parameters */
int nx2, ny2, nz2, nx, ny, nz;
logical timeron;

/* constants */
double tx1, tx2, tx3, ty1, ty2, ty3, tz1, tz2, tz3, 
       dx1, dx2, dx3, dx4, dx5, dy1, dy2, dy3, dy4, 
       dy5, dz1, dz2, dz3, dz4, dz5, dssp, dt, 
       ce[5][13], dxmax, dymax, dzmax, xxcon1, xxcon2, 
       xxcon3, xxcon4, xxcon5, dx1tx1, dx2tx1, dx3tx1,
       dx4tx1, dx5tx1, yycon1, yycon2, yycon3, yycon4,
       yycon5, dy1ty1, dy2ty1, dy3ty1, dy4ty1, dy5ty1,
       zzcon1, zzcon2, zzcon3, zzcon4, zzcon5, dz1tz1, 
       dz2tz1, dz3tz1, dz4tz1, dz5tz1, dnxm1, dnym1, 
       dnzm1, c1c2, c1c5, c3c4, c1345, conz1, c1, c2, 
       c3, c4, c5, c4dssp, c5dssp, dtdssp, dttx1, bt,
       dttx2, dtty1, dtty2, dttz1, dttz2, c2dttx1, 
       c2dtty1, c2dttz1, comz1, comz4, comz5, comz6, 
       c3c4tx3, c3c4ty3, c3c4tz3, c2iv, con43, con16;

/* main arrays */
double *u;
double *rhs;
double *forcing;

double *gpu_u;
double *gpu_rhs;
double *gpu_forcing;

double *gpu_us;
double *gpu_vs;
double *gpu_ws;
double *gpu_qs;
double *gpu_speed;
double *gpu_square;
double *gpu_rho_i;

/* tmp arrays lhs */
double lhs_ [P_SIZE][5];
double lhsp_[P_SIZE][5];
double lhsm_[P_SIZE][5];

int main(int argc, char *argv[])
{
    printf("\n Program started \n");
    
    int i, niter, step, result;
    double tmax;
    logical verified;
    const char *t_names[t_last + 1];

    timeron = inittrace(t_names);
    result = initparameters(argc, argv, &niter);
    if (result == 0)
        return -1;
    result = allocateArrays();
    if (result == 0)
        return -2;
    
    for (i = 1; i <= t_last; i++)
        timer_clear(i);

    // init
    set_constants();
    exact_rhs();
    initialize();

    // main loop
    timer_start(t_total);
    for (step = 1; step <= niter; step++) 
    {
        if ( (step % 20) == 0 || step == 1)
            printf(" Time step %4d\n", step);
        adi();
    }
    timer_stop(t_total);
    tmax = timer_read(t_total);

    verify(niter, &verified);
    print_results(niter, tmax, verified, t_names);
    
    result = deallocateArrays();
     if (result == 0)
        return -2;
    return 0;
}
