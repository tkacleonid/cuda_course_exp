#include "header.h"
#include <stdio.h>
#include <stdlib.h>

//---------------------------------------------------------------------
// compute the right hand side based on exact solution
//---------------------------------------------------------------------

void exact_rhs()
{
    double dtemp[5], xi, eta, zeta, dtpp, ue_[5][5], buf_[5][5], cuf_[5], q_[5];
    int m, i, j, k, p, ip1, im1, jp1, jm1, km1, kp1;

    for (k = 0; k <= nz - 1; k++) 
    {
        for (j = 0; j <= ny - 1; j++) 
        {
            for (i = 0; i <= nx - 1; i++) 
            {
                for (m = 0; m < 5; m++) 
                    forcing(k,j,i,m) = 0.0;
            }
        }
    }
        
    for (k = 1; k <= nz2; k++) 
    {
        for (j = 1; j <= ny2; j++) 
        {
            for (i = 1; i <= nx2; i++)
            {
                im1 = i - 1;
                ip1 = i + 1;

                for (p = i - 2; p <= i + 2; p++)
                {
                    eta = j * dnym1;
                    zeta = k * dnzm1;
                    xi = p * dnxm1;

                    exact_solution(xi, eta, zeta, dtemp);
                    dtpp = 1.0 / dtemp[0];
                    for (m = 0; m < 5; m++)
                    {
                        ue_[p - i + 2][m] = dtemp[m];
                        if (m != 0)
                            buf_[p - i + 2][m] = dtpp * dtemp[m];
                    }
                    cuf_[p - i + 2] = buf_[p - i + 2][1] * buf_[p - i + 2][1];
                    buf_[p - i + 2][0] = cuf_[p - i + 2] + buf_[p - i + 2][2] * buf_[p - i + 2][2] + buf_[p - i + 2][3] * buf_[p - i + 2][3];
                    q_[p - i + 2] = 0.5*(buf_[p - i + 2][1] * ue_[p - i + 2][1] + buf_[p - i + 2][2] * ue_[p - i + 2][2] + buf_[p - i + 2][3] * ue_[p - i + 2][3]);
                }

                forcing(k,j,i,0) = forcing(k,j,i,0) - tx2*(ue_[3][1] - ue_[1][1]) +
                    dx1tx1*(ue_[3][0] - 2.0 * ue_[2][0] + ue_[1][0]);

                forcing(k,j,i,1) = forcing(k,j,i,1) - tx2 * (
                    (ue_[3][1] * buf_[3][1] + c2*(ue_[3][4] - q_[3])) -
                    (ue_[1][1] * buf_[1][1] + c2*(ue_[1][4] - q_[1]))) +
                    xxcon1*(buf_[3][1] - 2.0*buf_[2][1] + buf_[1][1]) +
                    dx2tx1*(ue_[3][1] - 2.0* ue_[2][1] + ue_[1][1]);

                forcing(k,j,i,2) = forcing(k,j,i,2) - tx2 * (
                    ue_[3][2] * buf_[3][1] - ue_[1][2] * buf_[1][1]) +
                    xxcon2*(buf_[3][2] - 2.0*buf_[2][2] + buf_[1][2]) +
                    dx3tx1*(ue_[3][2] - 2.0*ue_[2][2] + ue_[1][2]);

                forcing(k,j,i,3) = forcing(k,j,i,3) - tx2*(
                    ue_[3][3] * buf_[3][1] - ue_[1][3] * buf_[1][1]) +
                    xxcon2*(buf_[3][3] - 2.0*buf_[2][3] + buf_[1][3]) +
                    dx4tx1*(ue_[3][3] - 2.0* ue_[2][3] + ue_[1][3]);

                forcing(k,j,i,4) = forcing(k,j,i,4) - tx2*(
                    buf_[3][1] * (c1*ue_[3][4] - c2*q_[3]) -
                    buf_[1][1] * (c1*ue_[1][4] - c2*q_[1])) +
                    0.5*xxcon3*(buf_[3][0] - 2.0*buf_[2][0] + buf_[1][0]) +
                    xxcon4*(cuf_[3] - 2.0*cuf_[2] + cuf_[1]) +
                    xxcon5*(buf_[3][4] - 2.0*buf_[2][4] + buf_[1][4]) +
                    dx5tx1*(ue_[3][4] - 2.0* ue_[2][4] + ue_[1][4]);

                if (i == 1)
                {                
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (5.0*ue_[2][m] - 4.0*ue_[3][m] + ue_[4][m]);
                }
                else if (i == 2)
                {                
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (-4.0*ue_[1][m] + 6.0*ue_[2][m] - 4.0*ue_[3][m] + ue_[4][m]);
                }
                else if (i == nx - 3)
                {                
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (ue_[0][m] - 4.0*ue_[1][m] + 6.0*ue_[2][m] - 4.0*ue_[3][m]);
                }
                else if (i == nx - 2)
                {             
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (ue_[0][m] - 4.0*ue_[1][m] + 5.0*ue_[2][m]);
                }
                else
                {
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (ue_[0][m] - 4.0*ue_[1][m] + 6.0*ue_[2][m] - 4.0*ue_[3][m] + ue_[4][m]);
                }


                jm1 = j - 1;
                jp1 = j + 1;

                for (p = j - 2; p <= j + 2; p++)
                {
                    eta = p * dnym1;
                    zeta = k * dnzm1;
                    xi = i * dnxm1;

                    exact_solution(xi, eta, zeta, dtemp);
                    dtpp = 1.0 / dtemp[0];          
                    for (m = 0; m < 5; m++)
                    {
                        ue_[p - j + 2][m] = dtemp[m];
                        if (m != 0)
                            buf_[p - j + 2][m] = dtpp * dtemp[m];
                    }
                    cuf_[p - j + 2] = buf_[p - j + 2][2] * buf_[p - j + 2][2];
                    buf_[p - j + 2][0] = cuf_[p - j + 2] + buf_[p - j + 2][1] * buf_[p - j + 2][1] + buf_[p - j + 2][3] * buf_[p - j + 2][3];
                    q_[p - j + 2] = 0.5*(buf_[p - j + 2][1] * ue_[p - j + 2][1] + buf_[p - j + 2][2] * ue_[p - j + 2][2] + buf_[p - j + 2][3] * ue_[p - j + 2][3]);
                }

                forcing(k,j,i,0) = forcing(k,j,i,0) -
                    ty2*(ue_[3][2] - ue_[1][2]) +
                    dy1ty1*(ue_[3][0] - 2.0*ue_[2][0] + ue_[1][0]);

                forcing(k,j,i,1) = forcing(k,j,i,1) - ty2*(
                    ue_[3][1] * buf_[3][2] - ue_[1][1] * buf_[1][2]) +
                    yycon2*(buf_[3][1] - 2.0*buf_[2][1] + buf_[1][1]) +
                    dy2ty1*(ue_[3][1] - 2.0* ue_[2][1] + ue_[1][1]);

                forcing(k,j,i,2) = forcing(k,j,i,2) - ty2*(
                    (ue_[3][2] * buf_[3][2] + c2*(ue_[3][4] - q_[3])) -
                    (ue_[1][2] * buf_[1][2] + c2*(ue_[1][4] - q_[1]))) +
                    yycon1*(buf_[3][2] - 2.0*buf_[2][2] + buf_[1][2]) +
                    dy3ty1*(ue_[3][2] - 2.0*ue_[2][2] + ue_[1][2]);

                forcing(k,j,i,3) = forcing(k,j,i,3) - ty2*(
                    ue_[3][3] * buf_[3][2] - ue_[1][3] * buf_[1][2]) +
                    yycon2*(buf_[3][3] - 2.0*buf_[2][3] + buf_[1][3]) +
                    dy4ty1*(ue_[3][3] - 2.0*ue_[2][3] + ue_[1][3]);

                forcing(k,j,i,4) = forcing(k,j,i,4) - ty2*(
                    buf_[3][2] * (c1*ue_[3][4] - c2*q_[3]) -
                    buf_[1][2] * (c1*ue_[1][4] - c2*q_[1])) +
                    0.5*yycon3*(buf_[3][0] - 2.0*buf_[2][0] +
                    buf_[1][0]) +
                    yycon4*(cuf_[3] - 2.0*cuf_[2] + cuf_[1]) +
                    yycon5*(buf_[3][4] - 2.0*buf_[2][4] + buf_[1][4]) +
                    dy5ty1*(ue_[3][4] - 2.0*ue_[2][4] + ue_[1][4]);

                if (j == 1)
                {                
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (5.0*ue_[2][m] - 4.0*ue_[3][m] + ue_[4][m]);
                }
                else if (j == 2)
                {                
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (-4.0*ue_[1][m] + 6.0*ue_[2][m] - 4.0*ue_[3][m] + ue_[4][m]);
                }
                else if (j == ny - 3)
                {                
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (ue_[0][m] - 4.0*ue_[1][m] + 6.0*ue_[2][m] - 4.0*ue_[3][m]);
                }
                else if (j == ny - 2)
                {            
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (ue_[0][m] - 4.0*ue_[1][m] + 5.0*ue_[2][m]);
                }
                else
                {
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (ue_[0][m] - 4.0*ue_[1][m] + 6.0*ue_[2][m] - 4.0*ue_[3][m] + ue_[4][m]);
                }

                km1 = k - 1;
                kp1 = k + 1;

                for (p = k - 2; p <= k + 2; p++)
                {
                    eta = j * dnym1;
                    zeta = p * dnzm1;
                    xi = i * dnxm1;

                    exact_solution(xi, eta, zeta, dtemp);
                    dtpp = 1.0 / dtemp[0];          
                    for (m = 0; m < 5; m++)
                    {
                        ue_[p - k + 2][m] = dtemp[m];
                        if (m != 0)
                            buf_[p - k + 2][m] = dtpp * dtemp[m];
                    }
                    cuf_[p - k + 2] = buf_[p - k + 2][3] * buf_[p - k + 2][3];
                    buf_[p - k + 2][0] = cuf_[p - k + 2] + buf_[p - k + 2][1] * buf_[p - k + 2][1] + buf_[p - k + 2][2] * buf_[p - k + 2][2];
                    q_[p - k + 2] = 0.5*(buf_[p - k + 2][1] * ue_[p - k + 2][1] + buf_[p - k + 2][2] * ue_[p - k + 2][2] + buf_[p - k + 2][3] * ue_[p - k + 2][3]);
                }

                forcing(k,j,i,0) = forcing(k,j,i,0) -
                    tz2*(ue_[3][3] - ue_[1][3]) +
                    dz1tz1*(ue_[3][0] - 2.0*ue_[2][0] + ue_[1][0]);

                forcing(k,j,i,1) = forcing(k,j,i,1) - tz2 * (
                    ue_[3][1] * buf_[3][3] - ue_[1][1] * buf_[1][3]) +
                    zzcon2*(buf_[3][1] - 2.0*buf_[2][1] + buf_[1][1]) +
                    dz2tz1*(ue_[3][1] - 2.0* ue_[2][1] + ue_[1][1]);

                forcing(k,j,i,2) = forcing(k,j,i,2) - tz2 * (
                    ue_[3][2] * buf_[3][3] - ue_[1][2] * buf_[1][3]) +
                    zzcon2*(buf_[3][2] - 2.0*buf_[2][2] + buf_[1][2]) +
                    dz3tz1*(ue_[3][2] - 2.0*ue_[2][2] + ue_[1][2]);

                forcing(k,j,i,3) = forcing(k,j,i,3) - tz2 * (
                    (ue_[3][3] * buf_[3][3] + c2*(ue_[3][4] - q_[3])) -
                    (ue_[1][3] * buf_[1][3] + c2*(ue_[1][4] - q_[1]))) +
                    zzcon1*(buf_[3][3] - 2.0*buf_[2][3] + buf_[1][3]) +
                    dz4tz1*(ue_[3][3] - 2.0*ue_[2][3] + ue_[1][3]);

                forcing(k,j,i,4) = forcing(k,j,i,4) - tz2 * (
                    buf_[3][3] * (c1*ue_[3][4] - c2*q_[3]) -
                    buf_[1][3] * (c1*ue_[1][4] - c2*q_[1])) +
                    0.5*zzcon3*(buf_[3][0] - 2.0*buf_[2][0] + buf_[1][0]) +
                    zzcon4*(cuf_[3] - 2.0*cuf_[2] + cuf_[1]) +
                    zzcon5*(buf_[3][4] - 2.0*buf_[2][4] + buf_[1][4]) +
                    dz5tz1*(ue_[3][4] - 2.0*ue_[2][4] + ue_[1][4]);

                if (k == 1)
                {                
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (5.0*ue_[2][m] - 4.0*ue_[3][m] + ue_[4][m]);
                }
                else if (k == 2)
                {                
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (-4.0*ue_[1][m] + 6.0*ue_[2][m] - 4.0*ue_[3][m] + ue_[4][m]);
                }
                else if (k == nz - 3)
                {                
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (ue_[0][m] - 4.0*ue_[1][m] + 6.0*ue_[2][m] - 4.0*ue_[3][m]);
                }
                else if (k == nz - 2)
                {                
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (ue_[0][m] - 4.0*ue_[1][m] + 5.0*ue_[2][m]);
                }
                else
                {
                    for (m = 0; m < 5; m++)
                        forcing(k,j,i,m) = forcing(k,j,i,m) - dssp * (ue_[0][m] - 4.0*ue_[1][m] + 6.0*ue_[2][m] - 4.0*ue_[3][m] + ue_[4][m]);
                }

                for (m = 0; m < 5; m++)
                    forcing(k,j,i,m) = -1.0 * forcing(k,j,i,m);
            }
        }
    }

}
