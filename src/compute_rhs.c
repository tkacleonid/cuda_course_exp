#include <math.h>
#include "header.h"

__global__ 
void rhs_init_kernel(double *u, double *rho_i, double *us, double *vs, 
      double *ws, double *square, double *qs, double *speed, const int nx, const int ny, const int nz, const double c1c2)
{
    double aux, rho_inv;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;
	if (i <= nx - 1 && j <= ny - 1 && k <= nz - 1) {
                rho_inv = 1.0 / u(k,j,i,0);
                rho_i(k,j,i) = rho_inv;
                us(k,j,i) = u(k,j,i,1) * rho_inv;
                vs(k,j,i) = u(k,j,i,2) * rho_inv;
                ws(k,j,i) = u(k,j,i,3) * rho_inv;
                square(k,j,i) = 0.5* (u(k,j,i,1) * u(k,j,i,1) + u(k,j,i,2) * u(k,j,i,2) + u(k,j,i,3) * u(k,j,i,3)) * rho_inv;
                qs(k,j,i) = square(k,j,i) * rho_inv;
                aux = c1c2*rho_inv* (u(k,j,i,4) - square(k,j,i));
                speed(k,j,i) = sqrt(aux);
	}
}

__global__ 
void rhs_mult_dt_kernel(double *rhs, const int nx2, const int ny2, const int nz2, const double dt)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    if (i <= nx2 && j <= ny2 && k <= nz2)
        for (int m = 0; m < 5; m++)
        	rhs(k,j,i,m) = rhs(k,j,i,m) * dt;

}

__global__ 
void rhs_x_kernel(double *u, double *rhs, double *rho_i, double *us, double *vs, 
      double *ws, double *square, double *qs, const int nx2, const int ny2, const int nz2, 
      const double dx1tx1, const double tx2, const double dx2tx1, const double xxcon2, const double con43, 
      const double dx3tx1, const double dx4tx1, const double dx5tx1, const double xxcon3, const double xxcon4, 
      const double xxcon5, const double c1, const double c2, const double dssp)
{
    double uijk, up1, um1;
    int m;
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    if (i <= nx2 && j <= ny2 && k <= nz2) {
                uijk = us(k,j,i);
                up1 = us(k,j,i + 1);
                um1 = us(k,j,i - 1);

                rhs(k,j,i,0) = rhs(k,j,i,0) + dx1tx1 *
                    (u(k,j,i + 1,0) - 2.0*u(k,j,i,0) + u(k,j,i - 1,0)) -
                    tx2 * (u(k,j,i + 1,1) - u(k,j,i - 1,1));

                rhs(k,j,i,1) = rhs(k,j,i,1) + dx2tx1 *
                    (u(k,j,i + 1,1) - 2.0*u(k,j,i,1) + u(k,j,i - 1,1)) +
                    xxcon2*con43 * (up1 - 2.0*uijk + um1) -
                    tx2 * (u(k,j,i + 1,1) * up1 - u(k,j,i - 1,1) * um1 +
                    (u(k,j,i + 1,4) - square(k,j,i + 1) -
                    u(k,j,i - 1,4) + square(k,j,i - 1)) * c2);

                rhs(k,j,i,2) = rhs(k,j,i,2) + dx3tx1 *
                    (u(k,j,i + 1,2) - 2.0*u(k,j,i,2) + u(k,j,i - 1,2)) +
                    xxcon2 * (vs(k,j,i + 1) - 2.0*vs(k,j,i) + vs(k,j,i - 1)) -
                    tx2 * (u(k,j,i + 1,2) * up1 - u(k,j,i - 1,2) * um1);

                rhs(k,j,i,3) = rhs(k,j,i,3) + dx4tx1 *
                    (u(k,j,i + 1,3) - 2.0*u(k,j,i,3) + u(k,j,i - 1,3)) +
                    xxcon2 * (ws(k,j,i + 1) - 2.0*ws(k,j,i) + ws(k,j,i - 1)) -
                    tx2 * (u(k,j,i + 1,3) * up1 - u(k,j,i - 1,3) * um1);

                rhs(k,j,i,4) = rhs(k,j,i,4) + dx5tx1 *
                    (u(k,j,i + 1,4) - 2.0*u(k,j,i,4) + u(k,j,i - 1,4)) +
                    xxcon3 * (qs(k,j,i + 1) - 2.0*qs(k,j,i) + qs(k,j,i - 1)) +
                    xxcon4 * (up1*up1 - 2.0*uijk*uijk + um1*um1) +
                    xxcon5 * (u(k,j,i + 1,4) * rho_i(k,j,i + 1) -
                    2.0*u(k,j,i,4) * rho_i(k,j,i) +
                    u(k,j,i - 1,4) * rho_i(k,j,i - 1)) -
                    tx2 * ((c1*u(k,j,i + 1,4) - c2*square(k,j,i + 1))*up1 -
                    (c1*u(k,j,i - 1,4) - c2*square(k,j,i - 1))*um1);
	}

    if (i <= nx2 && j <= ny2 && k <= nz2) {
                if (i == 1)
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (5.0*u(k,j,i,m) - 4.0*u(k,j,i + 1,m) + u(k,j,i + 2,m));
                }
                else if (i == 2)
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (-4.0*u(k,j,i - 1,m) + 6.0*u(k,j,i,m) - 4.0*u(k,j,i + 1,m) + u(k,j,i + 2,m));
                }
                else if (i == nx2 - 1)
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k,j,i - 2,m) - 4.0*u(k,j,i - 1,m) + 6.0*u(k,j,i,m) - 4.0*u(k,j,i + 1,m));
                }
                else if (i == nx2)
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k,j,i - 2,m) - 4.0*u(k,j,i - 1,m) + 5.0*u(k,j,i,m));
                }
                else
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k,j,i - 2,m) - 4.0*u(k,j,i - 1,m) + 6.0*u(k,j,i,m) - 4.0*u(k,j,i + 1,m) + u(k,j,i + 2,m));
                }
	}
}


__global__ 
void rhs_y_kernel(double *u, double *rhs, double *rho_i, double *us, double *vs, 
      double *ws, double *square, double *qs, const int nx2, const int ny2, const int nz2, 
      const double dy1ty1, const double ty2, const double dy2ty1, const double yycon2, const double con43, 
      const double dy3ty1, const double dy4ty1, const double dy5ty1, const double yycon3, const double yycon4, 
      const double yycon5, const double c1, const double c2, const double dssp)
{
    double vijk, vp1, vm1;
    int m;
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    if (i <= nx2 && j <= ny2 && k <= nz2) {
                vijk = vs(k,j,i);
                vp1 = vs(k,j + 1,i);
                vm1 = vs(k,j - 1,i);

                rhs(k,j,i,0) = rhs(k,j,i,0) + dy1ty1 *
                    (u(k,j + 1,i,0) - 2.0*u(k,j,i,0) + u(k,j - 1,i,0)) -
                    ty2 * (u(k,j + 1,i,2) - u(k,j - 1,i,2));

                rhs(k,j,i,1) = rhs(k,j,i,1) + dy2ty1 *
                    (u(k,j + 1,i,1) - 2.0*u(k,j,i,1) + u(k,j - 1,i,1)) +
                    yycon2 * (us(k,j + 1,i) - 2.0*us(k,j,i) + us(k,j - 1,i)) -
                    ty2 * (u(k,j + 1,i,1) * vp1 - u(k,j - 1,i,1) * vm1);

                rhs(k,j,i,2) = rhs(k,j,i,2) + dy3ty1 *
                    (u(k,j + 1,i,2) - 2.0*u(k,j,i,2) + u(k,j - 1,i,2)) +
                    yycon2*con43 * (vp1 - 2.0*vijk + vm1) -
                    ty2 * (u(k,j + 1,i,2) * vp1 - u(k,j - 1,i,2) * vm1 +
                    (u(k,j + 1,i,4) - square(k,j + 1,i) -
                    u(k,j - 1,i,4) + square(k,j - 1,i)) * c2);

                rhs(k,j,i,3) = rhs(k,j,i,3) + dy4ty1 *
                    (u(k,j + 1,i,3) - 2.0*u(k,j,i,3) + u(k,j - 1,i,3)) +
                    yycon2 * (ws(k,j + 1,i) - 2.0*ws(k,j,i) + ws(k,j - 1,i)) -
                    ty2 * (u(k,j + 1,i,3) * vp1 - u(k,j - 1,i,3) * vm1);

                rhs(k,j,i,4) = rhs(k,j,i,4) + dy5ty1 *
                    (u(k,j + 1,i,4) - 2.0*u(k,j,i,4) + u(k,j - 1,i,4)) +
                    yycon3 * (qs(k,j + 1,i) - 2.0*qs(k,j,i) + qs(k,j - 1,i)) +
                    yycon4 * (vp1*vp1 - 2.0*vijk*vijk + vm1*vm1) +
                    yycon5 * (u(k,j + 1,i,4) * rho_i(k,j + 1,i) -
                    2.0*u(k,j,i,4) * rho_i(k,j,i) +
                    u(k,j - 1,i,4) * rho_i(k,j - 1,i)) -
                    ty2 * ((c1*u(k,j + 1,i,4) - c2*square(k,j + 1,i)) * vp1 -
                    (c1*u(k,j - 1,i,4) - c2*square(k,j - 1,i)) * vm1);

                if (j == 1)
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (5.0*u(k,j,i,m) - 4.0*u(k,j + 1,i,m) + u(k,j + 2,i,m));
                }
                else if (j == 2)
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (-4.0*u(k,j - 1,i,m) + 6.0*u(k,j,i,m) - 4.0*u(k,j + 1,i,m) + u(k,j + 2,i,m));
                }
                else if (j == ny2 - 1)
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k,j - 2,i,m) - 4.0*u(k,j - 1,i,m) + 6.0*u(k,j,i,m) - 4.0*u(k,j + 1,i,m));
                }
                else if (j == ny2)
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k,j - 2,i,m) - 4.0*u(k,j - 1,i,m) + 5.0*u(k,j,i,m));
                }
                else
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k,j - 2,i,m) - 4.0*u(k,j - 1,i,m) + 6.0*u(k,j,i,m) - 4.0*u(k,j + 1,i,m) + u(k,j + 2,i,m));
                }
	}
}


__global__ 
void rhs_z_kernel(double *u, double *rhs, double *rho_i, double *us, double *vs, 
      double *ws, double *square, double *qs, const int nx2, const int ny2, const int nz2, 
      const double dz1tz1, const double tz2, const double dz2tz1, const double zzcon2, const double con43, 
      const double dz3tz1, const double dz4tz1, const double dz5tz1, const double zzcon3, const double zzcon4, 
      const double zzcon5, const double c1, const double c2, const double dssp)
{
    double wijk, wp1, wm1;
    int m;
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    if (i <= nx2 && j <= ny2 && k <= nz2) {
                wijk = ws(k,j,i);
                wp1 = ws(k + 1,j,i);
                wm1 = ws(k - 1,j,i);

                rhs(k,j,i,0) = rhs(k,j,i,0) + dz1tz1 *
                    (u(k + 1,j,i,0) - 2.0*u(k,j,i,0) + u(k - 1,j,i,0)) -
                    tz2 * (u(k + 1,j,i,3) - u(k - 1,j,i,3));

                rhs(k,j,i,1) = rhs(k,j,i,1) + dz2tz1 *
                    (u(k + 1,j,i,1) - 2.0*u(k,j,i,1) + u(k - 1,j,i,1)) +
                    zzcon2 * (us(k + 1,j,i) - 2.0*us(k,j,i) + us(k - 1,j,i)) -
                    tz2 * (u(k + 1,j,i,1) * wp1 - u(k - 1,j,i,1) * wm1);

                rhs(k,j,i,2) = rhs(k,j,i,2) + dz3tz1 *
                    (u(k + 1,j,i,2) - 2.0*u(k,j,i,2) + u(k - 1,j,i,2)) +
                    zzcon2 * (vs(k + 1,j,i) - 2.0*vs(k,j,i) + vs(k - 1,j,i)) -
                    tz2 * (u(k + 1,j,i,2) * wp1 - u(k - 1,j,i,2) * wm1);

                rhs(k,j,i,3) = rhs(k,j,i,3) + dz4tz1 *
                    (u(k + 1,j,i,3) - 2.0*u(k,j,i,3) + u(k - 1,j,i,3)) +
                    zzcon2*con43 * (wp1 - 2.0*wijk + wm1) -
                    tz2 * (u(k + 1,j,i,3) * wp1 - u(k - 1,j,i,3) * wm1 +
                    (u(k + 1,j,i,4) - square(k + 1,j,i) -
                    u(k - 1,j,i,4) + square(k - 1,j,i)) * c2);

                rhs(k,j,i,4) = rhs(k,j,i,4) + dz5tz1 *
                    (u(k + 1,j,i,4) - 2.0*u(k,j,i,4) + u(k - 1,j,i,4)) +
                    zzcon3 * (qs(k + 1,j,i) - 2.0*qs(k,j,i) + qs(k - 1,j,i)) +
                    zzcon4 * (wp1*wp1 - 2.0*wijk*wijk + wm1*wm1) +
                    zzcon5 * (u(k + 1,j,i,4) * rho_i(k + 1,j,i) -
                    2.0*u(k,j,i,4) * rho_i(k,j,i) +
                    u(k - 1,j,i,4) * rho_i(k - 1,j,i)) -
                    tz2 * ((c1*u(k + 1,j,i,4) - c2*square(k + 1,j,i))*wp1 -
                    (c1*u(k - 1,j,i,4) - c2*square(k - 1,j,i))*wm1);

                if (k == 1)
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (5.0*u(k,j,i,m) - 4.0*u(k + 1,j,i,m) + u(k + 2,j,i,m));
                }
                else if (k == 2)
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (-4.0*u(k - 1,j,i,m) + 6.0*u(k,j,i,m) - 4.0*u(k + 1,j,i,m) + u(k + 2,j,i,m));
                }
                else if (k == nz2 - 1)
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k - 2,j,i,m) - 4.0*u(k - 1,j,i,m) + 6.0*u(k,j,i,m) - 4.0*u(k + 1,j,i,m));
                }
                else if (k == nz2)
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k - 2,j,i,m) - 4.0*u(k - 1,j,i,m) + 5.0*u(k,j,i,m));
                }
                else
                {
                    for (m = 0; m < 5; m++)
                        rhs(k,j,i,m) = rhs(k,j,i,m) - dssp * (u(k - 2,j,i,m) - 4.0*u(k - 1,j,i,m) + 6.0*u(k,j,i,m) - 4.0*u(k + 1,j,i,m) + u(k + 2,j,i,m));
                }
	}
}

void compute_rhs(int rhs_verify)
{
    int i, j, k, m;

    size4 = sizeof(double) * nx * ny * nz * 5;
    size3 = sizeof(double) * nx * ny * nz;

    if (timeron) timer_start(t_rhs);
    
    rhs_init_kernel<<<dim3(nx/BS1 + 1, ny/BS2 + 1 , nz/BS3 + 1), dim3(BS1, BS2, BS3)>>>
            (g_u, g_rho_i, g_us, g_vs, g_ws, g_square, g_qs, g_speed, nx, ny, nz, c1c2);

    SAFE_CALL(cudaMemcpy(rho_i, g_rho_i, size3, cudaMemcpyDeviceToHost));    //Delete
    SAFE_CALL(cudaMemcpy(us, g_us, size3, cudaMemcpyDeviceToHost));    //Delete
    SAFE_CALL(cudaMemcpy(vs, g_vs, size3, cudaMemcpyDeviceToHost));    //Delete
    SAFE_CALL(cudaMemcpy(ws, g_ws, size3, cudaMemcpyDeviceToHost));    //Delete
    SAFE_CALL(cudaMemcpy(square, g_square, size3, cudaMemcpyDeviceToHost));    //Delete
    SAFE_CALL(cudaMemcpy(qs, g_qs, size3, cudaMemcpyDeviceToHost));    //Delete
    SAFE_CALL(cudaMemcpy(speed, g_speed, size3, cudaMemcpyDeviceToHost));    //Delete

    SAFE_CALL(cudaMemcpy(g_rhs, g_forcing, size4, cudaMemcpyDeviceToDevice)); //Useful

    if (timeron) timer_start(t_rhsx);
    rhs_x_kernel<<<dim3(nx2/BS1 + 1, ny2/BS2 + 1 , nz2/BS3 + 1), dim3(BS1, BS2, BS3)>>>
            (g_u, g_rhs, g_rho_i, g_us, g_vs, g_ws, g_square, g_qs, nx2, ny2, nz2, 
             dx1tx1, tx2, dx2tx1, xxcon2, con43, dx3tx1, dx4tx1, dx5tx1, xxcon3, xxcon4, xxcon5, c1, c2, dssp);
    if (timeron) timer_stop(t_rhsx);

    if (timeron) timer_start(t_rhsy);
    rhs_y_kernel<<<dim3(nx2/BS1 + 1, ny2/BS2 + 1 , nz2/BS3 + 1), dim3(BS1, BS2, BS3)>>>
            (g_u, g_rhs, g_rho_i, g_us, g_vs, g_ws, g_square, g_qs, nx2, ny2, nz2, 
             dy1ty1, ty2, dy2ty1, yycon2, con43, dy3ty1, dy4ty1, dy5ty1, yycon3, yycon4, yycon5, c1, c2, dssp);
    if (timeron) timer_stop(t_rhsy);

    if (timeron) timer_start(t_rhsz);
    rhs_z_kernel<<<dim3(nx2/BS1 + 1, ny2/BS2 + 1 , nz2/BS3 + 1), dim3(BS1, BS2, BS3)>>>
            (g_u, g_rhs, g_rho_i, g_us, g_vs, g_ws, g_square, g_qs, nx2, ny2, nz2, 
             dz1tz1, tz2, dz2tz1, zzcon2, con43, dz3tz1, dz4tz1, dz5tz1, zzcon3, zzcon4, zzcon5, c1, c2, dssp);
    if (timeron) timer_stop(t_rhsz);


    rhs_mult_dt_kernel<<<dim3(nx2/BS1 + 1, ny2/BS2 + 1 , nz2/BS3 + 1), dim3(BS1, BS2, BS3)>>>(g_rhs, nx2, ny2, nz2, dt);

    if (timeron) timer_stop(t_rhs);

	if(rhs_verify == 1) {
    	SAFE_CALL(cudaMemcpy(rhs, g_rhs, size4, cudaMemcpyDeviceToHost));  
    }

}
