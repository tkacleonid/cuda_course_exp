#include "header.h"

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the x-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the x-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------
void x_solve()
{
    int i, j, k, i1, i2, m;
    double ru1, rhon1, fac1, fac2;

    if (timeron) timer_start(t_xsolve);

    for (k = 1; k <= nz2; k++)
    {        
        for (j = 1; j <= ny2; j++)
        {
            for (m = 0; m < 5; m++)
            {
                lhs_ [0][m] = lhs_ [nx2 + 1][m] = 0.0;
                lhsp_[0][m] = lhsp_[nx2 + 1][m] = 0.0;
                lhsm_[0][m] = lhsm_[nx2 + 1][m] = 0.0;
            }

            lhs_ [0][2] = lhs_ [nx2 + 1][2] = 1.0;
            lhsp_[0][2] = lhsp_[nx2 + 1][2] = 1.0;
            lhsm_[0][2] = lhsm_[nx2 + 1][2] = 1.0;
                
            for (i = 1; i <= nx2; i++)
            {

            
                lhs_[i][0] = 0.0;
                ru1 = c3c4*rho_i[k][j][i - 1];
                rhon1 = max(max(dx2 + con43 * ru1, dx5 + c1c5 * ru1), max(dxmax + ru1, dx1));
                lhs_[i][1] = -dttx2 * us[k][j][i - 1] - dttx1 * rhon1;

                ru1 = c3c4*rho_i[k][j][i];
                rhon1 = max(max(dx2 + con43 * ru1, dx5 + c1c5 * ru1), max(dxmax + ru1, dx1));
                lhs_[i][2] = 1.0 + c2dttx1 * rhon1;

                ru1 = c3c4*rho_i[k][j][i + 1];
                rhon1 = max(max(dx2 + con43 * ru1, dx5 + c1c5 * ru1), max(dxmax + ru1, dx1));
                lhs_[i][3] = dttx2 * us[k][j][i + 1] - dttx1 * rhon1;
                lhs_[i][4] = 0.0;

                if (i == 1)
                {
                    lhs_[i][2] = lhs_[i][2] + comz5;
                    lhs_[i][3] = lhs_[i][3] - comz4;
                    lhs_[i][4] = lhs_[i][4] + comz1;
                }
                else if (i == 2)
                {
                    lhs_[i][1] = lhs_[i][1] - comz4;
                    lhs_[i][2] = lhs_[i][2] + comz6;
                    lhs_[i][3] = lhs_[i][3] - comz4;
                    lhs_[i][4] = lhs_[i][4] + comz1;
                }
                else if (i == nx - 3)
                {
                    lhs_[i][0] = lhs_[i][0] + comz1;
                    lhs_[i][1] = lhs_[i][1] - comz4;
                    lhs_[i][2] = lhs_[i][2] + comz6;
                    lhs_[i][3] = lhs_[i][3] - comz4;
                }
                else if (i == nx - 2)
                {
                    lhs_[i][0] = lhs_[i][0] + comz1;
                    lhs_[i][1] = lhs_[i][1] - comz4;
                    lhs_[i][2] = lhs_[i][2] + comz5;
                }
                else
                {
                    lhs_[i][0] = lhs_[i][0] + comz1;
                    lhs_[i][1] = lhs_[i][1] - comz4;
                    lhs_[i][2] = lhs_[i][2] + comz6;
                    lhs_[i][3] = lhs_[i][3] - comz4;
                    lhs_[i][4] = lhs_[i][4] + comz1;

                }

                lhsp_[i][0] = lhs_[i][0];
                lhsp_[i][1] = lhs_[i][1] - dttx2 * speed[k][j][i - 1];
                lhsp_[i][2] = lhs_[i][2];
                lhsp_[i][3] = lhs_[i][3] + dttx2 * speed[k][j][i + 1];
                lhsp_[i][4] = lhs_[i][4];

                lhsm_[i][0] = lhs_[i][0];
                lhsm_[i][1] = lhs_[i][1] + dttx2 * speed[k][j][i - 1];
                lhsm_[i][2] = lhs_[i][2];
                lhsm_[i][3] = lhs_[i][3] - dttx2 * speed[k][j][i + 1];
                lhsm_[i][4] = lhs_[i][4];
            }   
      
            for (i = 1; i <= nx2; i++)
            {
                i1 = i;
                i2 = i + 1;
                fac1 = 1.0 / lhs_[i - 1][2];
                lhs_[i - 1][3] = fac1 * lhs_[i - 1][3];
                lhs_[i - 1][4] = fac1 * lhs_[i - 1][4];
                for (m = 0; m < 3; m++)
                    rhs[k][j][i - 1][m] = fac1*rhs[k][j][i - 1][m];

                lhs_[i1][2] = lhs_[i1][2] - lhs_[i1][1] * lhs_[i - 1][3];
                lhs_[i1][3] = lhs_[i1][3] - lhs_[i1][1] * lhs_[i - 1][4];
                for (m = 0; m < 3; m++)
                    rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhs_[i1][1] * rhs[k][j][i - 1][m];

                lhs_[i2][1] = lhs_[i2][1] - lhs_[i2][0] * lhs_[i - 1][3];
                lhs_[i2][2] = lhs_[i2][2] - lhs_[i2][0] * lhs_[i - 1][4];
                for (m = 0; m < 3; m++)
                    rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhs_[i2][0] * rhs[k][j][i - 1][m];

                if (i == nx2)
                {
                    fac1 = 1.0 / lhs_[i1][2];
                    lhs_[i1][3] = fac1 * lhs_[i1][3];
                    lhs_[i1][4] = fac1 * lhs_[i1][4];
                    for (m = 0; m < 3; m++)
                        rhs[k][j][i1][m] = fac1 * rhs[k][j][i1][m];

                    lhs_[i2][2] = lhs_[i2][2] - lhs_[i2][1] * lhs_[i1][3];
                    lhs_[i2][3] = lhs_[i2][3] - lhs_[i2][1] * lhs_[i1][4];
                    for (m = 0; m < 3; m++)
                        rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhs_[i2][1] * rhs[k][j][i1][m];

                    fac2 = 1.0 / lhs_[i2][2];
                    for (m = 0; m < 3; m++)
                        rhs[k][j][i2][m] = fac2*rhs[k][j][i2][m];
                }
            
                m = 3;
                fac1 = 1.0 / lhsp_[i - 1][2];
                lhsp_[i - 1][3] = fac1 * lhsp_[i - 1][3];
                lhsp_[i - 1][4] = fac1 * lhsp_[i - 1][4];
                rhs[k][j][i - 1][m] = fac1 * rhs[k][j][i - 1][m];

                lhsp_[i1][2] = lhsp_[i1][2] - lhsp_[i1][1] * lhsp_[i - 1][3];
                lhsp_[i1][3] = lhsp_[i1][3] - lhsp_[i1][1] * lhsp_[i - 1][4];
                rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsp_[i1][1] * rhs[k][j][i - 1][m];

                lhsp_[i2][1] = lhsp_[i2][1] - lhsp_[i2][0] * lhsp_[i - 1][3];
                lhsp_[i2][2] = lhsp_[i2][2] - lhsp_[i2][0] * lhsp_[i - 1][4];
                rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsp_[i2][0] * rhs[k][j][i - 1][m];

                m = 4;
                fac1 = 1.0 / lhsm_[i - 1][2];
                lhsm_[i - 1][3] = fac1*lhsm_[i - 1][3];
                lhsm_[i - 1][4] = fac1*lhsm_[i - 1][4];
                rhs[k][j][i - 1][m] = fac1*rhs[k][j][i - 1][m];
                lhsm_[i1][2] = lhsm_[i1][2] - lhsm_[i1][1] * lhsm_[i - 1][3];
                lhsm_[i1][3] = lhsm_[i1][3] - lhsm_[i1][1] * lhsm_[i - 1][4];
                rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhsm_[i1][1] * rhs[k][j][i - 1][m];
                lhsm_[i2][1] = lhsm_[i2][1] - lhsm_[i2][0] * lhsm_[i - 1][3];
                lhsm_[i2][2] = lhsm_[i2][2] - lhsm_[i2][0] * lhsm_[i - 1][4];
                rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsm_[i2][0] * rhs[k][j][i - 1][m];

                if (i == nx2)
                {
                    m = 3;
                    fac1 = 1.0 / lhsp_[i1][2];
                    lhsp_[i1][3] = fac1 * lhsp_[i1][3];
                    lhsp_[i1][4] = fac1 * lhsp_[i1][4];
                    rhs[k][j][i1][m] = fac1 * rhs[k][j][i1][m];

                    lhsp_[i2][2] = lhsp_[i2][2] - lhsp_[i2][1] * lhsp_[i1][3];
                    lhsp_[i2][3] = lhsp_[i2][3] - lhsp_[i2][1] * lhsp_[i1][4];
                    rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsp_[i2][1] * rhs[k][j][i1][m];

                    m = 4;
                    fac1 = 1.0 / lhsm_[i1][2];
                    lhsm_[i1][3] = fac1 * lhsm_[i1][3];
                    lhsm_[i1][4] = fac1 * lhsm_[i1][4];
                    rhs[k][j][i1][m] = fac1*rhs[k][j][i1][m];

                    lhsm_[i2][2] = lhsm_[i2][2] - lhsm_[i2][1] * lhsm_[i1][3];
                    lhsm_[i2][3] = lhsm_[i2][3] - lhsm_[i2][1] * lhsm_[i1][4];
                    rhs[k][j][i2][m] = rhs[k][j][i2][m] - lhsm_[i2][1] * rhs[k][j][i1][m];

                    rhs[k][j][i2][3] = rhs[k][j][i2][3] / lhsp_[i2][2];
                    rhs[k][j][i2][4] = rhs[k][j][i2][4] / lhsm_[i2][2];

                    for (m = 0; m < 3; m++)
                        rhs[k][j][i1][m] = rhs[k][j][i1][m] - lhs_[i1][3] * rhs[k][j][i2][m];

                    rhs[k][j][i1][3] = rhs[k][j][i1][3] - lhsp_[i1][3] * rhs[k][j][i2][3];
                    rhs[k][j][i1][4] = rhs[k][j][i1][4] - lhsm_[i1][3] * rhs[k][j][i2][4];
                }
            }
     
            for (i = nx2; i >= 1; i--)
            {
                i1 = i;
                i2 = i + 1;
                for (m = 0; m < 3; m++)
                    rhs[k][j][i - 1][m] = rhs[k][j][i - 1][m] - lhs_[i - 1][3] * rhs[k][j][i1][m] - lhs_[i - 1][4] * rhs[k][j][i2][m];

                rhs[k][j][i - 1][3] = rhs[k][j][i - 1][3] - lhsp_[i - 1][3] * rhs[k][j][i1][3] - lhsp_[i - 1][4] * rhs[k][j][i2][3];
                rhs[k][j][i - 1][4] = rhs[k][j][i - 1][4] - lhsm_[i - 1][3] * rhs[k][j][i1][4] - lhsm_[i - 1][4] * rhs[k][j][i2][4];
            }
        }
    }

    //---------------------------------------------------------------------
    // Do the block-diagonal inversion          
    //---------------------------------------------------------------------
    double r1, r2, r3, r4, r5, t1, t2;
    if (timeron) timer_start(t_ninvr);

    for (k = 1; k <= nz2; k++)
    {
        for (j = 1; j <= ny2; j++)
        {
            for (i = 1; i <= nx2; i++)
            {
                r1 = rhs[k][j][i][0];
                r2 = rhs[k][j][i][1];
                r3 = rhs[k][j][i][2];
                r4 = rhs[k][j][i][3];
                r5 = rhs[k][j][i][4];

                t1 = bt * r3;
                t2 = 0.5 * (r4 + r5);

                rhs[k][j][i][0] = -r2;
                rhs[k][j][i][1] = r1;
                rhs[k][j][i][2] = bt * (r4 - r5);
                rhs[k][j][i][3] = -t1 + t2;
                rhs[k][j][i][4] = t1 + t2;
            }
        }
    }
    if (timeron) timer_stop(t_ninvr);
    if (timeron) timer_stop(t_xsolve);
}
