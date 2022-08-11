
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_blas.h>


const double pi = 4*atan(1);


int main(){
    double a [] = {0.109818, 0.405771, 2.22776}; // 13.0073, 1.962079, 0.444529, 0.1219492
    size_t n = sizeof(a)/sizeof(a[0]);

    gsl_vector * alpha = gsl_vector_alloc(n);
    for(int i = 0; i < n; i++){
        gsl_vector_set(alpha, i, a[i]);
    }

    gsl_matrix * S = gsl_matrix_alloc(n,n);
    for(int i = 0; i < n; i++){
        for(int j = i; j < n; j++){
            gsl_matrix_set(S, i, j, pow(pi/(gsl_vector_get(alpha,i)+gsl_vector_get(alpha,j)), 1.5));
            gsl_matrix_set(S, j, i, gsl_matrix_get(S,i,j));
        }
    }

    gsl_matrix * H = gsl_matrix_alloc(n,n);
    for(int i = 0; i < n; i++){
        for(int j = i; j < n; j++){
            gsl_matrix_set(H, i, j, (3*gsl_vector_get(alpha,i)*gsl_vector_get(alpha,j)*pow(pi/(gsl_vector_get(alpha,i)+gsl_vector_get(alpha,j)), 1.5)-2*pi)/(gsl_vector_get(alpha,i)+gsl_vector_get(alpha,j)));
            gsl_matrix_set(H, j, i, gsl_matrix_get(H,i,j));
        }
    }

    gsl_vector * eval = gsl_vector_alloc(n);
    gsl_matrix * evec = gsl_matrix_alloc(n,n);
    gsl_eigen_symmv_workspace * w = gsl_eigen_symmv_alloc(n);


    // Diagonalize S
    gsl_eigen_symmv(S, eval, evec, w);

    // Create matrix V
    gsl_matrix * V = gsl_matrix_alloc(n,n);
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            gsl_matrix_set(V, i, j, gsl_matrix_get(evec, i, j) / sqrt(gsl_vector_get(eval,j)));
        }
    }

    // Create matrix H1
    gsl_matrix * HV = gsl_matrix_alloc(n,n);
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, H, V, 0.0, HV);
    gsl_matrix * H1 = gsl_matrix_alloc(n,n);
    gsl_blas_dgemm(CblasTrans, CblasNoTrans, 1.0, V, HV, 0.0, H1);
    gsl_matrix_free(HV);

    // Diagonalize H1
    gsl_eigen_symmv(H1, eval, evec, w);
    gsl_vector_fprintf(stdout,eval,"%.8f");

    // Compute eigenvectors of H from the ones of H1
    gsl_blas_dgemm(CblasNoTrans, CblasNoTrans, 1.0, V, evec, 0.0, H1);
    gsl_matrix_fprintf(stdout,H1,"%.8f");

    return 0;
}
