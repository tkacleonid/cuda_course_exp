#include "header.h"
#include <stdio.h>
#include <stdlib.h>

//---------------------------------------------------------------------
// This subroutine initializes the field variable u using 
// tri-linear transfinite interpolation of the boundary values     
//---------------------------------------------------------------------
void initialize()
{
    double Pface[2][3][5], Pxi, Peta, Pzeta, temp[5];

    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                u(k, j, i, 0) = 1.0;
                u(k, j, i, 1) = 0.0;
                u(k, j, i, 2) = 0.0;
                u(k, j, i, 3) = 0.0;
                u(k, j, i, 4) = 1.0;

                double xi = i * dnxm1;
                double eta = j * dnym1;
                double zeta = k * dnzm1;

                for (int ix = 0; ix < 2; ++ix) {
                    Pxi = ix;
                    exact_solution(Pxi, eta, zeta, &Pface[ix][0][0]);
                }

                for (int iy = 0; iy < 2; ++iy) {
                    Peta = iy;
                    exact_solution(xi, Peta, zeta, &Pface[iy][1][0]);
                }

                for (int iz = 0; iz < 2; ++iz) {
                    Pzeta = iz;
                    exact_solution(xi, eta, Pzeta, &Pface[iz][2][0]);
                }

                for (int m = 0; m < 5; ++m) {
                    Pxi = xi * Pface[1][0][m] + (1.0 - xi) * Pface[0][0][m];
                    Peta = eta * Pface[1][1][m] + (1.0 - eta) * Pface[0][1][m];
                    Pzeta = zeta * Pface[1][2][m] + (1.0 - zeta) * Pface[0][2][m];

                    u(k, j, i, m) = Pxi + Peta + Pzeta - Pxi*Peta - Pxi*Pzeta - Peta*Pzeta + Pxi*Peta*Pzeta;
                }

                if (i == 0) {
                    xi = 0.0;
                    eta = j * dnym1;
                    zeta = k * dnzm1;

                    exact_solution(xi, eta, zeta, temp);

                    for (int m = 0; m < 5; ++m)
                        u(k, j, i, m) = temp[m];
                } else if (i == nx - 1) {
                    xi = 1.0;
                    eta = j * dnym1;
                    zeta = k * dnzm1;

                    exact_solution(xi, eta, zeta, temp);

                    for (int m = 0; m < 5; ++m)
                        u(k, j, i, m) = temp[m];
                } else if (j == 0) {
                    xi = i * dnxm1;
                    eta = 0.0;
                    zeta = k * dnzm1;

                    exact_solution(xi, eta, zeta, temp);

                    for (int m = 0; m < 5; ++m)
                        u(k, j, i, m) = temp[m];
                } else if (j == ny - 1) {
                    xi = i * dnxm1;
                    eta = 1.0;
                    zeta = k * dnzm1;

                    exact_solution(xi, eta, zeta, temp);

                    for (int m = 0; m < 5; ++m)
                        u(k, j, i, m) = temp[m];
                } else if (k == 0) {
                    xi = i * dnxm1;
                    eta = j * dnym1;
                    zeta = 0.0;

                    exact_solution(xi, eta, zeta, temp);

                    for (int m = 0; m < 5; ++m)
                        u(k, j, i, m) = temp[m];
                } else if (k == nz - 1) {
                    xi = i * dnxm1;
                    eta = j * dnym1;
                    zeta = 1.0;

                    exact_solution(xi, eta, zeta, temp);

                    for (int m = 0; m < 5; ++m)
                        u(k, j, i, m) = temp[m];
                }
            }
        }
    }

}

logical inittrace(const char** t_names)
{
    logical timeron = false;
    if (TRACE)
    {
        timeron = true;
        t_names[t_total] = "total";
        t_names[t_rhsx] = "x direction";
        t_names[t_rhsy] = "y direction";
        t_names[t_rhsz] = "z direction";
        t_names[t_rhs] = "rhs";
        t_names[t_xsolve] = "x_solve";
        t_names[t_ysolve] = "y_solve";
        t_names[t_zsolve] = "z_solve";
        t_names[t_tzetar] = "block-diag mv";
        t_names[t_ninvr] = "block-diag inv";
        t_names[t_pinvr] = "block-diag inv";
        t_names[t_txinvr] = "x invr";
        t_names[t_add] = "add";
    }
    return timeron;
}

int initparameters(int argc, char **argv, int *niter)
{
    int OK = true;

    FILE *fp;
    if ((fp = fopen("inpudata", "r")) != NULL)
    {
        int result = 0;
        printf(" Reading from input file inpudata\n");
        result = fscanf(fp, "%d", niter);
        while (fgetc(fp) != '\n');
        result = fscanf(fp, "%lf", &dt);
        while (fgetc(fp) != '\n');
        result = fscanf(fp, "%d%d%d", &nx, &ny, &nz);
        fclose(fp);
    }
    else
    {
        //printf(" No input file inpudata. Using compiled defaults\n");
        *niter = NITER_DEFAULT;
        dt = DT_DEFAULT;
        nx = P_SIZE;
        ny = P_SIZE;
        nz = P_SIZE;
    }

    printf(" Size: %4dx%4dx%4d\n", nx, ny, nz);
    printf(" Iterations: %4d    dt: %10.6f\n", *niter, dt);
    printf("\n");

    if ((nx > P_SIZE) || (ny > P_SIZE) || (nz > P_SIZE))
    {
        printf(" %d, %d, %d\n", nx, ny, nz);
        printf(" Problem size too big for compiled array sizes\n");
        OK = false;
    }
    else
    {
        nx2 = nx - 2;
        ny2 = ny - 2;
        nz2 = nz - 2;
    }

    return OK;
}

int allocateArrays()
{
    size4 = sizeof(double) * nx * ny * nz * 5;
    size3 = sizeof(double) * nx * ny * nz;
    size2 = sizeof(double) * nx * 5;

    u = (double *) malloc(size4);
    rhs = (double *) malloc(size4);
    forcing = (double *) malloc(size4);

    us = (double *) malloc(size3);
    vs = (double *) malloc(size3);
    ws = (double *) malloc(size3);
    qs = (double *) malloc(size3);    
    rho_i = (double *) malloc(size3);    
    speed = (double *) malloc(size3);    
    square = (double *) malloc(size3);   
    
    lhs_ = (double *) malloc(size4);    
    lhsp_ = (double *) malloc(size4);    
    lhsm_ = (double *) malloc(size4);    

    SAFE_CALL(cudaMalloc(&g_u, size4));
    SAFE_CALL(cudaMalloc(&g_rhs, size4));
    SAFE_CALL(cudaMalloc(&g_forcing, size4));

    SAFE_CALL(cudaMalloc(&g_us, size3));
    SAFE_CALL(cudaMalloc(&g_vs, size3));
    SAFE_CALL(cudaMalloc(&g_ws, size3));
    SAFE_CALL(cudaMalloc(&g_qs, size3));
    SAFE_CALL(cudaMalloc(&g_rho_i, size3));
    SAFE_CALL(cudaMalloc(&g_speed, size3));
    SAFE_CALL(cudaMalloc(&g_square, size3));

    SAFE_CALL(cudaMalloc(&g_lhs_, size4));
    SAFE_CALL(cudaMalloc(&g_lhsp_, size4));
    SAFE_CALL(cudaMalloc(&g_lhsm_, size4));
    
    return 1;
}

int deallocateArrays()
{
    free(u);
    free(rhs);
    free(forcing);

    free(us);
    free(vs);
    free(ws);
    free(qs);
    free(rho_i);
    free(speed);
    free(square);

    free(lhs_);
    free(lhsp_);
    free(lhsm_);

    SAFE_CALL(cudaFree(g_u));
    SAFE_CALL(cudaFree(g_rhs));
    SAFE_CALL(cudaFree(g_forcing));

    SAFE_CALL(cudaFree(g_us));
    SAFE_CALL(cudaFree(g_vs));
    SAFE_CALL(cudaFree(g_ws));
    SAFE_CALL(cudaFree(g_qs));
    SAFE_CALL(cudaFree(g_rho_i));
    SAFE_CALL(cudaFree(g_speed));
    SAFE_CALL(cudaFree(g_square));

    SAFE_CALL(cudaFree(g_lhs_));
    SAFE_CALL(cudaFree(g_lhsp_));
    SAFE_CALL(cudaFree(g_lhsm_));

    return 1;
}
