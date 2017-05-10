#include "header.h"

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the y-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the y-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
#define lhs_(j, m) lhs_[(m) * P_SIZE * P_SIZE * P_SIZE + (j) * P_SIZE * P_SIZE + (k) * P_SIZE + i]
#define lhsp_(j, m) lhsp_[(m) * P_SIZE * P_SIZE * P_SIZE + (j) * P_SIZE * P_SIZE + (k) * P_SIZE + i]
#define lhsm_(j, m) lhsm_[(m) * P_SIZE * P_SIZE * P_SIZE + (j) * P_SIZE * P_SIZE + (k) * P_SIZE + i]

void y_solve()
{
    int i, j, k, j1, j2, m;
    double ru1, rhoq1, fac1, fac2;

    if (timeron) timer_start(t_ysolve);

    for (k = 1; k <= nz2; k++)
    {        
        for (i = 1; i <= nx2; i++)
        {
            for (m = 0; m < 5; m++)
            {
                lhs_(0,m) = lhs_(ny2 + 1,m) = 0.0;
                lhsp_(0,m) = lhsp_(ny2 + 1,m) = 0.0;
                lhsm_(0,m) = lhsm_(ny2 + 1,m) = 0.0;
            }
            lhs_(0,2) = lhs_(ny2 + 1,2) = 1.0;
            lhsp_(0,2) = lhsp_(ny2 + 1,2) = 1.0;
            lhsm_(0,2) = lhsm_(ny2 + 1,2) = 1.0;

            for (j = 1; j <= ny2; j++)
            {
                lhs_(j,0) = 0.0;

                ru1 = c3c4*rho_i(k,j - 1,i);
                rhoq1 = max(max(dy3 + con43*ru1, dy5 + c1c5*ru1), max(dymax + ru1, dy1));
                lhs_(j,1) = -dtty2 * vs(k,j - 1,i) - dtty1 * rhoq1;

                ru1 = c3c4*rho_i(k,j,i);
                rhoq1 = max(max(dy3 + con43*ru1, dy5 + c1c5*ru1), max(dymax + ru1, dy1));
                lhs_(j,2) = 1.0 + c2dtty1 * rhoq1;

                ru1 = c3c4*rho_i(k,j + 1,i);
                rhoq1 = max(max(dy3 + con43*ru1, dy5 + c1c5*ru1), max(dymax + ru1, dy1));
                lhs_(j,3) = dtty2 * vs(k,j + 1,i) - dtty1 * rhoq1;
                lhs_(j,4) = 0.0;

                if (j == 1)
                {
                    lhs_(j,2) = lhs_(j,2) + comz5;
                    lhs_(j,3) = lhs_(j,3) - comz4;
                    lhs_(j,4) = lhs_(j,4) + comz1;
                }
                else if (j == 2)
                {
                    lhs_(j,1) = lhs_(j,1) - comz4;
                    lhs_(j,2) = lhs_(j,2) + comz6;
                    lhs_(j,3) = lhs_(j,3) - comz4;
                    lhs_(j,4) = lhs_(j,4) + comz1;
                }
                else if (j == ny - 3)
                {
                    lhs_(j,0) = lhs_(j,0) + comz1;
                    lhs_(j,1) = lhs_(j,1) - comz4;
                    lhs_(j,2) = lhs_(j,2) + comz6;
                    lhs_(j,3) = lhs_(j,3) - comz4;
                }
                else if (j == ny - 2)
                {
                    lhs_(j,0) = lhs_(j,0) + comz1;
                    lhs_(j,1) = lhs_(j,1) - comz4;
                    lhs_(j,2) = lhs_(j,2) + comz5;
                }
                else
                {
                    lhs_(j,0) = lhs_(j,0) + comz1;
                    lhs_(j,1) = lhs_(j,1) - comz4;
                    lhs_(j,2) = lhs_(j,2) + comz6;
                    lhs_(j,3) = lhs_(j,3) - comz4;
                    lhs_(j,4) = lhs_(j,4) + comz1;
                }

                lhsp_(j,0) = lhs_(j,0);
                lhsp_(j,1) = lhs_(j,1) - dtty2 * speed(k,j - 1,i);
                lhsp_(j,2) = lhs_(j,2);
                lhsp_(j,3) = lhs_(j,3) + dtty2 * speed(k,j + 1,i);
                lhsp_(j,4) = lhs_(j,4);

                lhsm_(j,0) = lhs_(j,0);
                lhsm_(j,1) = lhs_(j,1) + dtty2 * speed(k,j - 1,i);
                lhsm_(j,2) = lhs_(j,2);
                lhsm_(j,3) = lhs_(j,3) - dtty2 * speed(k,j + 1,i);
                lhsm_(j,4) = lhs_(j,4);
            }

            for (j = 1; j <= ny2; j++)
            {
                j1 = j;
                j2 = j + 1;

                fac1 = 1.0 / lhs_(j - 1,2);
                lhs_(j - 1,3) = fac1*lhs_(j - 1,3);
                lhs_(j - 1,4) = fac1*lhs_(j - 1,4);
                for (m = 0; m < 3; m++)
                    rhs(k,j - 1,i,m) = fac1*rhs(k,j - 1,i,m);

                lhs_(j1,2) = lhs_(j1,2) - lhs_(j1,1) * lhs_(j - 1,3);
                lhs_(j1,3) = lhs_(j1,3) - lhs_(j1,1) * lhs_(j - 1,4);
                for (m = 0; m < 3; m++)
                    rhs(k,j1,i,m) = rhs(k,j1,i,m) - lhs_(j1,1) * rhs(k,j - 1,i,m);

                lhs_(j2,1) = lhs_(j2,1) - lhs_(j2,0) * lhs_(j - 1,3);
                lhs_(j2,2) = lhs_(j2,2) - lhs_(j2,0) * lhs_(j - 1,4);
                for (m = 0; m < 3; m++)
                    rhs(k,j2,i,m) = rhs(k,j2,i,m) - lhs_(j2,0) * rhs(k,j - 1,i,m);

                if (j == ny2)
                {
                    fac1 = 1.0 / lhs_(j1,2);
                    lhs_(j1,3) = fac1 * lhs_(j1,3);
                    lhs_(j1,4) = fac1 * lhs_(j1,4);
                    for (m = 0; m < 3; m++)
                        rhs(k,j1,i,m) = fac1 * rhs(k,j1,i,m);

                    lhs_(j2,2) = lhs_(j2,2) - lhs_(j2,1) * lhs_(j1,3);
                    lhs_(j2,3) = lhs_(j2,3) - lhs_(j2,1) * lhs_(j1,4);
                    for (m = 0; m < 3; m++)
                        rhs(k,j2,i,m) = rhs(k,j2,i,m) - lhs_(j2,1) * rhs(k,j1,i,m);

                    fac2 = 1.0 / lhs_(j2,2);
                    for (m = 0; m < 3; m++)
                        rhs(k,j2,i,m) = fac2 * rhs(k,j2,i,m);
                }
            
                m = 3;
                fac1 = 1.0 / lhsp_(j - 1,2);
                lhsp_(j - 1,3) = fac1 * lhsp_(j - 1,3);
                lhsp_(j - 1,4) = fac1 * lhsp_(j - 1,4);

                rhs(k,j - 1,i,m) = fac1 * rhs(k,j - 1,i,m);
                lhsp_(j1,2) = lhsp_(j1,2) - lhsp_(j1,1) * lhsp_(j - 1,3);
                lhsp_(j1,3) = lhsp_(j1,3) - lhsp_(j1,1) * lhsp_(j - 1,4);

                rhs(k,j1,i,m) = rhs(k,j1,i,m) - lhsp_(j1,1) * rhs(k,j - 1,i,m);
                lhsp_(j2,1) = lhsp_(j2,1) - lhsp_(j2,0) * lhsp_(j - 1,3);
                lhsp_(j2,2) = lhsp_(j2,2) - lhsp_(j2,0) * lhsp_(j - 1,4);
                rhs(k,j2,i,m) = rhs(k,j2,i,m) - lhsp_(j2,0) * rhs(k,j - 1,i,m);

                m = 4;
                fac1 = 1.0 / lhsm_(j - 1,2);
                lhsm_(j - 1,3) = fac1 * lhsm_(j - 1,3);
                lhsm_(j - 1,4) = fac1 * lhsm_(j - 1,4);

                rhs(k,j - 1,i,m) = fac1 * rhs(k,j - 1,i,m);
                lhsm_(j1,2) = lhsm_(j1,2) - lhsm_(j1,1) * lhsm_(j - 1,3);
                lhsm_(j1,3) = lhsm_(j1,3) - lhsm_(j1,1) * lhsm_(j - 1,4);

                rhs(k,j1,i,m) = rhs(k,j1,i,m) - lhsm_(j1,1) * rhs(k,j - 1,i,m);
                lhsm_(j2,1) = lhsm_(j2,1) - lhsm_(j2,0) * lhsm_(j - 1,3);
                lhsm_(j2,2) = lhsm_(j2,2) - lhsm_(j2,0) * lhsm_(j - 1,4);
                rhs(k,j2,i,m) = rhs(k,j2,i,m) - lhsm_(j2,0) * rhs(k,j - 1,i,m);

                if (j == ny2)
                {
                    m = 3;
                    fac1 = 1.0 / lhsp_(j1,2);
                    lhsp_(j1,3) = fac1 * lhsp_(j1,3);
                    lhsp_(j1,4) = fac1 * lhsp_(j1,4);

                    rhs(k,j1,i,m) = fac1 * rhs(k,j1,i,m);
                    lhsp_(j2,2) = lhsp_(j2,2) - lhsp_(j2,1) * lhsp_(j1,3);
                    lhsp_(j2,3) = lhsp_(j2,3) - lhsp_(j2,1) * lhsp_(j1,4);
                    rhs(k,j2,i,m) = rhs(k,j2,i,m) - lhsp_(j2,1) * rhs(k,j1,i,m);

                    m = 4;
                    fac1 = 1.0 / lhsm_(j1,2);
                    lhsm_(j1,3) = fac1 * lhsm_(j1,3);
                    lhsm_(j1,4) = fac1 * lhsm_(j1,4);
                    rhs(k,j1,i,m) = fac1 * rhs(k,j1,i,m);

                    lhsm_(j2,2) = lhsm_(j2,2) - lhsm_(j2,1) * lhsm_(j1,3);
                    lhsm_(j2,3) = lhsm_(j2,3) - lhsm_(j2,1) * lhsm_(j1,4);
                    rhs(k,j2,i,m) = rhs(k,j2,i,m) - lhsm_(j2,1) * rhs(k,j1,i,m);

                    rhs(k,j2,i,3) = rhs(k,j2,i,3) / lhsp_(j2,2);
                    rhs(k,j2,i,4) = rhs(k,j2,i,4) / lhsm_(j2,2);

                    for (m = 0; m < 3; m++)
                        rhs(k,j1,i,m) = rhs(k,j1,i,m) - lhs_(j1,3) * rhs(k,j2,i,m);
                    rhs(k,j1,i,3) = rhs(k,j1,i,3) - lhsp_(j1,3) * rhs(k,j2,i,3);
                    rhs(k,j1,i,4) = rhs(k,j1,i,4) - lhsm_(j,3) * rhs(k,j2,i,4);
                }
            }


            for (j = ny2; j >= 1; j--)
            {
                j1 = j;
                j2 = j + 1;

                for (m = 0; m < 3; m++)
                    rhs(k,j - 1,i,m) = rhs(k,j - 1,i,m) - lhs_(j - 1,3) * rhs(k,j1,i,m) - lhs_(j - 1,4) * rhs(k,j2,i,m);

                rhs(k,j - 1,i,3) = rhs(k,j - 1,i,3) - lhsp_(j - 1,3) * rhs(k,j1,i,3) - lhsp_(j - 1,4) * rhs(k,j2,i,3);
                rhs(k,j - 1,i,4) = rhs(k,j - 1,i,4) - lhsm_(j - 1,3) * rhs(k,j1,i,4) - lhsm_(j - 1,4) * rhs(k,j2,i,4);
            }
        }
    }

    //---------------------------------------------------------------------
    // block-diagonal matrix-vector multiplication                       
    //---------------------------------------------------------------------
    double r1, r2, r3, r4, r5, t1, t2;
    if (timeron) timer_start(t_pinvr);

    for (k = 1; k <= nz2; k++)
    {
        for (j = 1; j <= ny2; j++)
        {
            for (i = 1; i <= nx2; i++)
            {
                r1 = rhs(k,j,i,0);
                r2 = rhs(k,j,i,1);
                r3 = rhs(k,j,i,2);
                r4 = rhs(k,j,i,3);
                r5 = rhs(k,j,i,4);

                t1 = bt * r1;
                t2 = 0.5 * (r4 + r5);

                rhs(k,j,i,0) = bt * (r4 - r5);
                rhs(k,j,i,1) = -r3;
                rhs(k,j,i,2) = r2;
                rhs(k,j,i,3) = -t1 + t2;
                rhs(k,j,i,4) = t1 + t2;
            }
        }
    }
    if (timeron) timer_stop(t_pinvr);
    if (timeron) timer_stop(t_ysolve);
}
#undef lhs_
#undef lhsp_
#undef lhsm_
