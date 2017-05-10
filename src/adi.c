#include "header.h"

#include <stdio.h>

//---------------------------------------------------------------------
// block-diagonal matrix-vector multiplication                  
//---------------------------------------------------------------------

/*
#define us(i, j, k) us[(i) * P_SIZE * P_SIZE + (j) * P_SIZE + k]
#define vs(i, j, k) vs[(i) * P_SIZE * P_SIZE + (j) * P_SIZE + k]
#define ws(i, j, k) ws[(i) * P_SIZE * P_SIZE + (j) * P_SIZE + k]
#define qs(i, j, k) qs[(i) * P_SIZE * P_SIZE + (j) * P_SIZE + k]
#define rho_i(i, j, k) rho_i[(i) * P_SIZE * P_SIZE + (j) * P_SIZE + k]
#define speed(i, j, k) speed[(i) * P_SIZE * P_SIZE + (j) * P_SIZE + k]

#define u(i, j, k, m) u[(i) * P_SIZE * P_SIZE * P_SIZE + (j) * P_SIZE * P_SIZE + (k) * P_SIZE + m]
#define rhs(i, j, k, m) rhs[(i) * P_SIZE * P_SIZE * P_SIZE + (j) * P_SIZE * P_SIZE + (k) * P_SIZE + m]

*/



__global__ void xinvr_kernel(const double *rho_i, const double *speed, const double *us, 
                             const double *vs, const double *ws, const double *qs,
                             double *rhs, const int nx2, const int ny2, const int nz2, const double bt, const double c2)
{
    
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;
    
    double t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv, qq;
    
    if (i <= nx2 && j <= ny2 && k <= nz2)
    {
        ru1 = rho_i(k,j,i);
        uu = us(k,j,i);
        vv = vs(k,j,i);
        ww = ws(k,j,i);
        ac = speed(k,j,i);
        qq = qs(k,j,i);
        ac2inv = ac*ac;

        r1 = rhs(k,j,i,0);
        r2 = rhs(k,j,i,1);
        r3 = rhs(k,j,i,2);
        r4 = rhs(k,j,i,3);
        r5 = rhs(k,j,i,4);

        t1 = c2 / ac2inv * ( qq * r1 - uu * r2 - vv * r3 - ww * r4 + r5);
        t2 = bt * ru1 * (uu * r1 - r2);
        t3 = (bt * ru1 * ac) * t1;

        rhs(k,j,i,0) = r1 - t1;
        rhs(k,j,i,1) = -ru1 * (ww*r1 - r4);
        rhs(k,j,i,2) = ru1 * (vv*r1 - r3);
        rhs(k,j,i,3) = -t2 + t3;
        rhs(k,j,i,4) = t2 + t3;
    }
}

void xinvr2()
{

    size4 = sizeof(double) * nx * ny * nz * 5;
    size3 = sizeof(double) * nx * ny * nz;

    if (timeron) timer_start(t_txinvr);

    xinvr_kernel<<<dim3(nx2/BS1 + 1, ny2/BS2 + 1 , nz2/BS3 + 1), dim3(BS1, BS2, BS3)>>>
          (g_rho_i, g_speed, g_us, g_vs, g_ws, g_qs, g_rhs, nx2, ny2, nz2, bt, c2);

    SAFE_CALL(cudaMemcpy(rhs, g_rhs, size4, cudaMemcpyDeviceToHost));

    if (timeron) timer_stop(t_txinvr);
}




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



__global__ 
void add_kernel(double *u, double *rhs, const int nx2, const int ny2, const int nz2)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    if (i <= nx2 && j <= ny2 && k <= nz2)
        for (int m = 0; m < 5; m++)
            u(k,j,i,m) = u(k,j,i,m) + rhs(k,j,i,m);

}

void add()
{

    if (timeron) timer_start(t_add);

    SAFE_CALL(cudaMemcpy(g_u, u, size4, cudaMemcpyHostToDevice));
    SAFE_CALL(cudaMemcpy(g_rhs, rhs, size4, cudaMemcpyHostToDevice));
    
    add_kernel<<<dim3(nx2/BS1 + 1, ny2/BS2 + 1 , nz2/BS3 + 1), dim3(BS1, BS2, BS3)>>>(g_u, g_rhs, nx2, ny2, nz2);

    SAFE_CALL(cudaMemcpy(u, g_u, size4, cudaMemcpyDeviceToHost));


    if (timeron) timer_stop(t_add);
}

/*
#undef us
#undef vs
#undef ws
#undef qs
#undef rho_i
#undef speed

#undef u
#undef rhs

*/

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
    xinvr2();
    x_solve();
    y_solve();
    z_solve();
    add();
}