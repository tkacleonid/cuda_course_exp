#include "header.h"

#include <stdio.h>

//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication                  
//---------------------------------------------------------------------



void xinvr()
{
    int i, j, k;
    double t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv;

    if (timeron) timer_start(t_txinvr);

    for (k = 1; k <= nz2; k++)
    {
        for (j = 1; j <= ny2; j++)
        {            
            for (i = 1; i <= nx2; i++)
            {
                ru1 = rho_i(k,j,i);
                uu = us(k,j,i);
                vv = vs(k,j,i);
                ww = ws(k,j,i);
                ac = speed(k,j,i);
                ac2inv = ac*ac;

                r1 = rhs(k,j,i,0);
                r2 = rhs(k,j,i,1);
                r3 = rhs(k,j,i,2);
                r4 = rhs(k,j,i,3);
                r5 = rhs(k,j,i,4);

                t1 = c2 / ac2inv * (qs(k,j,i) * r1 - uu*r2 - vv*r3 - ww*r4 + r5);
                t2 = bt * ru1 * (uu * r1 - r2);
                t3 = (bt * ru1 * ac) * t1;

                rhs(k,j,i,0) = r1 - t1;
                rhs(k,j,i,1) = -ru1 * (ww*r1 - r4);
                rhs(k,j,i,2) = ru1 * (vv*r1 - r3);
                rhs(k,j,i,3) = -t2 + t3;
                rhs(k,j,i,4) = t2 + t3;
            }
        }
    }
    if (timeron) timer_stop(t_txinvr);
}


void add2()
{
    int i, j, k, m;
    if (timeron) timer_start(t_add);

    for (k = 1; k <= nz2; k++) {
        for (j = 1; j <= ny2; j++) {
            for (i = 1; i <= nx2; i++) {
                for (m = 0; m < 5; m++) {
                    u(k,j,i,m) = u(k,j,i,m) + rhs(k,j,i,m);
                }
            }
        }
    }

    if (timeron) timer_stop(t_add);
}


void adi()
{
    compute_rhs(0);
    xinvr();
    x_solve();
    y_solve();
    z_solve();
    add2();
}