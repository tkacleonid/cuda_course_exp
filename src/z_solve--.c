#include "header.h"

//---------------------------------------------------------------------
// this function performs the solution of the approximate factorization
// step in the z-direction for all five matrix components
// simultaneously. The Thomas algorithm is employed to solve the
// systems for the z-lines. Boundary conditions are non-periodic
//---------------------------------------------------------------------

#define lhs_(k, m) lhs_[(m) * P_SIZE * P_SIZE * P_SIZE + (k) * P_SIZE * P_SIZE + (j) * P_SIZE + i]
#define lhsp_(k, m) lhsp_[(m) * P_SIZE * P_SIZE * P_SIZE + (k) * P_SIZE * P_SIZE + (j) * P_SIZE + i]
#define lhsm_(k, m) lhsm_[(m) * P_SIZE * P_SIZE * P_SIZE + (k) * P_SIZE * P_SIZE + (j) * P_SIZE + i]

__global__
void z_solve_init_kernel(double *lhs_, double *lhsp_, double *lhsm_,  double *ws,  double *rho_i, double *speed,  
             const int nx2, const int ny2, const int nz2, const int nz, 
             const double c3c4, const double dz4, const double con43, const double dz5, const double c1c5, 
             const double dzmax, const double dz1, const double dttz2, const double dttz1, const double c2dttz1, 
             const double comz1, const double comz4, const double comz5, const double comz6)
{
    int m;
    double ru1, rhos1;

    int i = threadIdx.x + blockIdx.x * blockDim.x ;
    int j = threadIdx.y + blockIdx.y * blockDim.y ;
    int k = threadIdx.z + blockIdx.z * blockDim.z ;

    if ( (k == 0 || k == nz2 + 1) && j <= ny2 && i <= nx2) {
            for (m = 0; m < 5; m++)
            {
                lhs_ (k,m) =  0.0;
                lhsp_(k,m) =  0.0;
                lhsm_(k,m) =  0.0;
            }
            lhs_ (k,2) = 1.0;
            lhsp_(k,2) = 1.0;
            lhsm_(k,2) = 1.0;

	} else 
    if (i <= nx2 && j <= ny2 && k <= nz2)
	{
            
                lhs_(k,0) = 0.0;

                ru1 = c3c4*rho_i(k - 1,j,i);
                rhos1 = max(max(dz4 + con43*ru1, dz5 + c1c5*ru1), max(dzmax + ru1, dz1));
                lhs_(k,1) = -dttz2 * ws(k - 1,j,i) - dttz1 * rhos1;

                ru1 = c3c4*rho_i(k,j,i);
                rhos1 = max(max(dz4 + con43*ru1, dz5 + c1c5*ru1), max(dzmax + ru1, dz1));
                lhs_(k,2) = 1.0 + c2dttz1 * rhos1;

                ru1 = c3c4*rho_i(k + 1,j,i);
                rhos1 = max(max(dz4 + con43*ru1, dz5 + c1c5*ru1), max(dzmax + ru1, dz1));
                lhs_(k,3) = dttz2 * ws(k + 1,j,i) - dttz1 * rhos1;
                lhs_(k,4) = 0.0;

                if (k == 1)
                {
                    lhs_(k,2) = lhs_(k,2) + comz5;
                    lhs_(k,3) = lhs_(k,3) - comz4;
                    lhs_(k,4) = lhs_(k,4) + comz1;
                }
                else if (k == 2)
                {
                    lhs_(k,1) = lhs_(k,1) - comz4;
                    lhs_(k,2) = lhs_(k,2) + comz6;
                    lhs_(k,3) = lhs_(k,3) - comz4;
                    lhs_(k,4) = lhs_(k,4) + comz1;
                }
                else if (k == nz2 - 1)
                {
                    lhs_(k,0) = lhs_(k,0) + comz1;
                    lhs_(k,1) = lhs_(k,1) - comz4;
                    lhs_(k,2) = lhs_(k,2) + comz6;
                    lhs_(k,3) = lhs_(k,3) - comz4;
                }
                else if (k == nz2)
                {
                    lhs_(k,0) = lhs_(k,0) + comz1;
                    lhs_(k,1) = lhs_(k,1) - comz4;
                    lhs_(k,2) = lhs_(k,2) + comz5;
                }
                else
                {
                    lhs_(k,0) = lhs_(k,0) + comz1;
                    lhs_(k,1) = lhs_(k,1) - comz4;
                    lhs_(k,2) = lhs_(k,2) + comz6;
                    lhs_(k,3) = lhs_(k,3) - comz4;
                    lhs_(k,4) = lhs_(k,4) + comz1;
                }

                lhsp_(k,0) = lhs_(k,0);
                lhsp_(k,1) = lhs_(k,1) - dttz2 * speed(k - 1,j,i);
                lhsp_(k,2) = lhs_(k,2);
                lhsp_(k,3) = lhs_(k,3) + dttz2 * speed(k + 1,j,i);
                lhsp_(k,4) = lhs_(k,4);
                lhsm_(k,0) = lhs_(k,0);
                lhsm_(k,1) = lhs_(k,1) + dttz2 * speed(k - 1,j,i);
                lhsm_(k,2) = lhs_(k,2);
                lhsm_(k,3) = lhs_(k,3) - dttz2 * speed(k + 1,j,i);
                lhsm_(k,4) = lhs_(k,4);

    }
}  


__global__
void z_solve_loop_kernel(double *rhs, double *lhs_, double *lhsp_, double *lhsm_, 
             const int nx2, const int ny2, const int nz2)
{
     
    int k, k1, k2, m;
    double fac1, fac2;

    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;

    if (j <= ny2 && i <= nx2)
	{  

            for (k = 1; k <= nz2; k++)
            {
                k1 = k;
                k2 = k + 1;

                fac1 = 1.0 / lhs_(k - 1,2);
                lhs_(k - 1,3) = fac1 * lhs_(k - 1,3);
                lhs_(k - 1,4) = fac1 * lhs_(k - 1,4);
                for (m = 0; m < 3; m++)
                    rhs(k - 1,j,i,m) = fac1 * rhs(k - 1,j,i,m);

                lhs_(k1,2) = lhs_(k1,2) - lhs_(k1,1) * lhs_(k - 1,3);
                lhs_(k1,3) = lhs_(k1,3) - lhs_(k1,1) * lhs_(k - 1,4);
                for (m = 0; m < 3; m++)
                    rhs(k1,j,i,m) = rhs(k1,j,i,m) - lhs_(k1,1) * rhs(k - 1,j,i,m);

                lhs_(k2,1) = lhs_(k2,1) - lhs_(k2,0) * lhs_(k - 1,3);
                lhs_(k2,2) = lhs_(k2,2) - lhs_(k2,0) * lhs_(k - 1,4);
                for (m = 0; m < 3; m++)
                    rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhs_(k2,0) * rhs(k - 1,j,i,m);

                if (k == nz2)
                {
                    fac1 = 1.0 / lhs_(k1,2);
                    lhs_(k1,3) = fac1 * lhs_(k1,3);
                    lhs_(k1,4) = fac1 * lhs_(k1,4);
                    for (m = 0; m < 3; m++)
                        rhs(k1,j,i,m) = fac1 * rhs(k1,j,i,m);

                    lhs_(k2,2) = lhs_(k2,2) - lhs_(k2,1) * lhs_(k1,3);
                    lhs_(k2,3) = lhs_(k2,3) - lhs_(k2,1) * lhs_(k1,4);
                    for (m = 0; m < 3; m++)
                        rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhs_(k2,1) * rhs(k1,j,i,m);

                    fac2 = 1.0 / lhs_(k2,2);
                    for (m = 0; m < 3; m++)
                        rhs(k2,j,i,m) = fac2 * rhs(k2,j,i,m);
                }

                m = 3;
                fac1 = 1.0 / lhsp_(k - 1,2);
                lhsp_(k - 1,3) = fac1 * lhsp_(k - 1,3);
                lhsp_(k - 1,4) = fac1 * lhsp_(k - 1,4);
                rhs(k - 1,j,i,m) = fac1 * rhs(k - 1,j,i,m);

                lhsp_(k1,2) = lhsp_(k1,2) - lhsp_(k1,1) * lhsp_(k - 1,3);
                lhsp_(k1,3) = lhsp_(k1,3) - lhsp_(k1,1) * lhsp_(k - 1,4);
                rhs(k1,j,i,m) = rhs(k1,j,i,m) - lhsp_(k1,1) * rhs(k - 1,j,i,m);

                lhsp_(k2,1) = lhsp_(k2,1) - lhsp_(k2,0) * lhsp_(k - 1,3);
                lhsp_(k2,2) = lhsp_(k2,2) - lhsp_(k2,0) * lhsp_(k - 1,4);
                rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhsp_(k2,0) * rhs(k - 1,j,i,m);

                m = 4;
                fac1 = 1.0 / lhsm_(k - 1,2);
                lhsm_(k - 1,3) = fac1 * lhsm_(k - 1,3);
                lhsm_(k - 1,4) = fac1 * lhsm_(k - 1,4);
                rhs(k - 1,j,i,m) = fac1 * rhs(k - 1,j,i,m);

                lhsm_(k1,2) = lhsm_(k1,2) - lhsm_(k1,1) * lhsm_(k - 1,3);
                lhsm_(k1,3) = lhsm_(k1,3) - lhsm_(k1,1) * lhsm_(k - 1,4);
                rhs(k1,j,i,m) = rhs(k1,j,i,m) - lhsm_(k1,1) * rhs(k - 1,j,i,m);

                lhsm_(k2,1) = lhsm_(k2,1) - lhsm_(k2,0) * lhsm_(k - 1,3);
                lhsm_(k2,2) = lhsm_(k2,2) - lhsm_(k2,0) * lhsm_(k - 1,4);
                rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhsm_(k2,0) * rhs(k - 1,j,i,m);

                if (k == nz2)
                {
                    m = 3;
                    fac1 = 1.0 / lhsp_(k1,2);
                    lhsp_(k1,3) = fac1 * lhsp_(k1,3);
                    lhsp_(k1,4) = fac1 * lhsp_(k1,4);
                    rhs(k1,j,i,m) = fac1 * rhs(k1,j,i,m);

                    lhsp_(k2,2) = lhsp_(k2,2) - lhsp_(k2,1) * lhsp_(k1,3);
                    lhsp_(k2,3) = lhsp_(k2,3) - lhsp_(k2,1) * lhsp_(k1,4);
                    rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhsp_(k2,1) * rhs(k1,j,i,m);

                    m = 4;
                    fac1 = 1.0 / lhsm_(k1,2);
                    lhsm_(k1,3) = fac1 * lhsm_(k1,3);
                    lhsm_(k1,4) = fac1 * lhsm_(k1,4);
                    rhs(k1,j,i,m) = fac1 * rhs(k1,j,i,m);

                    lhsm_(k2,2) = lhsm_(k2,2) - lhsm_(k2,1) * lhsm_(k1,3);
                    lhsm_(k2,3) = lhsm_(k2,3) - lhsm_(k2,1) * lhsm_(k1,4);
                    rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhsm_(k2,1) * rhs(k1,j,i,m);

                    rhs(k2,j,i,3) = rhs(k2,j,i,3) / lhsp_(k2,2);
                    rhs(k2,j,i,4) = rhs(k2,j,i,4) / lhsm_(k2,2);

                    for (m = 0; m < 3; m++)
                        rhs(k1,j,i,m) = rhs(k1,j,i,m) - lhs_(k1,3) * rhs(k2,j,i,m);

                    rhs(k1,j,i,3) = rhs(k1,j,i,3) - lhsp_(k1,3) * rhs(k2,j,i,3);
                    rhs(k1,j,i,4) = rhs(k1,j,i,4) - lhsm_(k1,3) * rhs(k2,j,i,4);
                }
            }

            for (k = nz2; k >= 1; k--)
            {
                k1 = k;
                k2 = k + 1;

                for (m = 0; m < 3; m++)
                    rhs(k - 1,j,i,m) = rhs(k - 1,j,i,m) - lhs_(k - 1,3) * rhs(k1,j,i,m) - lhs_(k - 1,4) * rhs(k2,j,i,m);

                rhs(k - 1,j,i,3) = rhs(k - 1,j,i,3) - lhsp_(k - 1,3) * rhs(k1,j,i,3) - lhsp_(k - 1,4) * rhs(k2,j,i,3);
                rhs(k - 1,j,i,4) = rhs(k - 1,j,i,4) - lhsm_(k - 1,3) * rhs(k1,j,i,4) - lhsm_(k - 1,4) * rhs(k2,j,i,4);
            }

	}
}

__global__
void z_solve_invr_kernel(double *rhs, double *u, double *us, double *vs, double *ws, double *qs, double *speed, 
                         const int nx2, const int ny2, const int nz2, const double bt, const double c2iv)
{
    double t1, t2, t3, ac, xvel, yvel, zvel;
    double btuz, ac2u, uzik1, r1, r2, r3, r4, r5;

    int i = threadIdx.x + blockIdx.x * blockDim.x + 1;
    int j = threadIdx.y + blockIdx.y * blockDim.y + 1;
    int k = threadIdx.z + blockIdx.z * blockDim.z + 1;

    if (i <= nx2 && j <= ny2 && k <= nz2)
	{
                xvel = us(k,j,i);
                yvel = vs(k,j,i);
                zvel = ws(k,j,i);
                ac = speed(k,j,i);

                ac2u = ac*ac;

                r1 = rhs(k,j,i,0);
                r2 = rhs(k,j,i,1);
                r3 = rhs(k,j,i,2);
                r4 = rhs(k,j,i,3);
                r5 = rhs(k,j,i,4);

                uzik1 = u(k,j,i,0);
                btuz = bt * uzik1;

                t1 = btuz / ac * (r4 + r5);
                t2 = r3 + t1;
                t3 = btuz * (r4 - r5);

                rhs(k,j,i,0) = t2;
                rhs(k,j,i,1) = -uzik1*r2 + xvel*t2;
                rhs(k,j,i,2) = uzik1*r1 + yvel*t2;
                rhs(k,j,i,3) = zvel*t2 + t3;
                rhs(k,j,i,4) = uzik1*(-xvel*r2 + yvel*r1) + qs(k,j,i) * t2 + c2iv*ac2u*t1 + zvel*t3;
	}
}


void z_solve()
{
    int i, j, k, k1, k2, m;
    double fac1, fac2;

    if (timeron) timer_start(t_zsolve);

    SAFE_CALL(cudaMemcpy(g_rhs, rhs, size4, cudaMemcpyHostToDevice)); //Delete

	z_solve_init_kernel<<<dim3(nx/BS1 + 1, ny/BS2 + 1 , nz/BS3 + 1), dim3(BS1, BS2, BS3)>>>
             (g_lhs_, g_lhsp_, g_lhsm_,  g_ws,  g_rho_i, g_speed, nx2, ny2, nz2, nz, 
              c3c4, dz4, con43, dz5, c1c5, dzmax, dz1, dttz2, dttz1, c2dttz1,comz1, comz4, comz5, comz6);


	z_solve_loop_kernel<<<dim3(nx2/BS1 + 1, ny2/BS2 + 1 , 1), dim3(BS1, BS2, 1)>>>
             (g_rhs, g_lhs_, g_lhsp_, g_lhsm_, nx2, ny2, nz2);

/*
	SAFE_CALL(cudaMemcpy(rhs, g_rhs, size4, cudaMemcpyDeviceToHost)); 

	SAFE_CALL(cudaMemcpy(lhs_, g_lhs_, size4, cudaMemcpyDeviceToHost)); 
	SAFE_CALL(cudaMemcpy(lhsp_, g_lhsp_, size4, cudaMemcpyDeviceToHost));  
	SAFE_CALL(cudaMemcpy(lhsm_, g_lhsm_, size4, cudaMemcpyDeviceToHost)); 

    for (j = 1; j <= ny2; j++) 
    {
        for (i = 1; i <= nx2; i++)
        {
            for (k = 1; k <= nz2; k++)
            {
                k1 = k;
                k2 = k + 1;

                fac1 = 1.0 / lhs_(k - 1,2);
                lhs_(k - 1,3) = fac1 * lhs_(k - 1,3);
                lhs_(k - 1,4) = fac1 * lhs_(k - 1,4);
                for (m = 0; m < 3; m++)
                    rhs(k - 1,j,i,m) = fac1 * rhs(k - 1,j,i,m);

                lhs_(k1,2) = lhs_(k1,2) - lhs_(k1,1) * lhs_(k - 1,3);
                lhs_(k1,3) = lhs_(k1,3) - lhs_(k1,1) * lhs_(k - 1,4);
                for (m = 0; m < 3; m++)
                    rhs(k1,j,i,m) = rhs(k1,j,i,m) - lhs_(k1,1) * rhs(k - 1,j,i,m);

                lhs_(k2,1) = lhs_(k2,1) - lhs_(k2,0) * lhs_(k - 1,3);
                lhs_(k2,2) = lhs_(k2,2) - lhs_(k2,0) * lhs_(k - 1,4);
                for (m = 0; m < 3; m++)
                    rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhs_(k2,0) * rhs(k - 1,j,i,m);

                if (k == nz2)
                {
                    fac1 = 1.0 / lhs_(k1,2);
                    lhs_(k1,3) = fac1 * lhs_(k1,3);
                    lhs_(k1,4) = fac1 * lhs_(k1,4);
                    for (m = 0; m < 3; m++)
                        rhs(k1,j,i,m) = fac1 * rhs(k1,j,i,m);

                    lhs_(k2,2) = lhs_(k2,2) - lhs_(k2,1) * lhs_(k1,3);
                    lhs_(k2,3) = lhs_(k2,3) - lhs_(k2,1) * lhs_(k1,4);
                    for (m = 0; m < 3; m++)
                        rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhs_(k2,1) * rhs(k1,j,i,m);

                    fac2 = 1.0 / lhs_(k2,2);
                    for (m = 0; m < 3; m++)
                        rhs(k2,j,i,m) = fac2 * rhs(k2,j,i,m);
                }

                m = 3;
                fac1 = 1.0 / lhsp_(k - 1,2);
                lhsp_(k - 1,3) = fac1 * lhsp_(k - 1,3);
                lhsp_(k - 1,4) = fac1 * lhsp_(k - 1,4);
                rhs(k - 1,j,i,m) = fac1 * rhs(k - 1,j,i,m);

                lhsp_(k1,2) = lhsp_(k1,2) - lhsp_(k1,1) * lhsp_(k - 1,3);
                lhsp_(k1,3) = lhsp_(k1,3) - lhsp_(k1,1) * lhsp_(k - 1,4);
                rhs(k1,j,i,m) = rhs(k1,j,i,m) - lhsp_(k1,1) * rhs(k - 1,j,i,m);

                lhsp_(k2,1) = lhsp_(k2,1) - lhsp_(k2,0) * lhsp_(k - 1,3);
                lhsp_(k2,2) = lhsp_(k2,2) - lhsp_(k2,0) * lhsp_(k - 1,4);
                rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhsp_(k2,0) * rhs(k - 1,j,i,m);

                m = 4;
                fac1 = 1.0 / lhsm_(k - 1,2);
                lhsm_(k - 1,3) = fac1 * lhsm_(k - 1,3);
                lhsm_(k - 1,4) = fac1 * lhsm_(k - 1,4);
                rhs(k - 1,j,i,m) = fac1 * rhs(k - 1,j,i,m);

                lhsm_(k1,2) = lhsm_(k1,2) - lhsm_(k1,1) * lhsm_(k - 1,3);
                lhsm_(k1,3) = lhsm_(k1,3) - lhsm_(k1,1) * lhsm_(k - 1,4);
                rhs(k1,j,i,m) = rhs(k1,j,i,m) - lhsm_(k1,1) * rhs(k - 1,j,i,m);

                lhsm_(k2,1) = lhsm_(k2,1) - lhsm_(k2,0) * lhsm_(k - 1,3);
                lhsm_(k2,2) = lhsm_(k2,2) - lhsm_(k2,0) * lhsm_(k - 1,4);
                rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhsm_(k2,0) * rhs(k - 1,j,i,m);

                if (k == nz2)
                {
                    m = 3;
                    fac1 = 1.0 / lhsp_(k1,2);
                    lhsp_(k1,3) = fac1 * lhsp_(k1,3);
                    lhsp_(k1,4) = fac1 * lhsp_(k1,4);
                    rhs(k1,j,i,m) = fac1 * rhs(k1,j,i,m);

                    lhsp_(k2,2) = lhsp_(k2,2) - lhsp_(k2,1) * lhsp_(k1,3);
                    lhsp_(k2,3) = lhsp_(k2,3) - lhsp_(k2,1) * lhsp_(k1,4);
                    rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhsp_(k2,1) * rhs(k1,j,i,m);

                    m = 4;
                    fac1 = 1.0 / lhsm_(k1,2);
                    lhsm_(k1,3) = fac1 * lhsm_(k1,3);
                    lhsm_(k1,4) = fac1 * lhsm_(k1,4);
                    rhs(k1,j,i,m) = fac1 * rhs(k1,j,i,m);

                    lhsm_(k2,2) = lhsm_(k2,2) - lhsm_(k2,1) * lhsm_(k1,3);
                    lhsm_(k2,3) = lhsm_(k2,3) - lhsm_(k2,1) * lhsm_(k1,4);
                    rhs(k2,j,i,m) = rhs(k2,j,i,m) - lhsm_(k2,1) * rhs(k1,j,i,m);

                    rhs(k2,j,i,3) = rhs(k2,j,i,3) / lhsp_(k2,2);
                    rhs(k2,j,i,4) = rhs(k2,j,i,4) / lhsm_(k2,2);

                    for (m = 0; m < 3; m++)
                        rhs(k1,j,i,m) = rhs(k1,j,i,m) - lhs_(k1,3) * rhs(k2,j,i,m);

                    rhs(k1,j,i,3) = rhs(k1,j,i,3) - lhsp_(k1,3) * rhs(k2,j,i,3);
                    rhs(k1,j,i,4) = rhs(k1,j,i,4) - lhsm_(k1,3) * rhs(k2,j,i,4);
                }
            }

            for (k = nz2; k >= 1; k--)
            {
                k1 = k;
                k2 = k + 1;

                for (m = 0; m < 3; m++)
                    rhs(k - 1,j,i,m) = rhs(k - 1,j,i,m) - lhs_(k - 1,3) * rhs(k1,j,i,m) - lhs_(k - 1,4) * rhs(k2,j,i,m);

                rhs(k - 1,j,i,3) = rhs(k - 1,j,i,3) - lhsp_(k - 1,3) * rhs(k1,j,i,3) - lhsp_(k - 1,4) * rhs(k2,j,i,3);
                rhs(k - 1,j,i,4) = rhs(k - 1,j,i,4) - lhsm_(k - 1,3) * rhs(k1,j,i,4) - lhsm_(k - 1,4) * rhs(k2,j,i,4);
            }
        }
    }
*/
    //---------------------------------------------------------------------
    // block-diagonal matrix-vector multiplication                       
    //---------------------------------------------------------------------

    if (timeron) timer_start(t_tzetar);

 //   SAFE_CALL(cudaMemcpy(g_rhs, rhs, size4, cudaMemcpyHostToDevice)); //Delete

	z_solve_invr_kernel<<<dim3(nx2/BS1 + 1, ny2/BS2 + 1, nz2/BS3 + 1), dim3(BS1, BS2, BS3)>>> 
                        (g_rhs, g_u, g_us, g_vs, g_ws, g_qs, g_speed, nx2, ny2, nz2, bt, c2iv);

	SAFE_CALL(cudaMemcpy(rhs, g_rhs, size4, cudaMemcpyDeviceToHost)); 

    if (timeron) timer_stop(t_tzetar);
    if (timeron) timer_stop(t_zsolve);
}
#undef lhs_
#undef lhsp_
#undef lhsm_