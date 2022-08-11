
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_multimin.h >


const double pi = 4*atan(1);

/*
I calculate the best expansion on a set of Gaussians with given widths alpha.
I return the minimum eigenvalue, which represents an estimate of the binding energy.
*/
double bestBindingEnergy(const gsl_vector * alpha){
    size_t n = alpha->size;

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
    gsl_eigen_gensymmv_workspace * w =  gsl_eigen_gensymmv_alloc(n);
    gsl_eigen_gensymmv(H, S, eval, evec, w);

    //gsl_matrix_fprintf(stdout,evec,"%f");

    return gsl_vector_min(eval);
}


int main(){
    double alpha [] = {0.109818, 0.405771, 2.22776}; // 13.0073, 1.962079, 0.444529, 0.1219492
    size_t n = sizeof(alpha)/sizeof(alpha[0]);

    gsl_vector * alpha_gsl = gsl_vector_alloc(n);
    for(int i = 0; i < n; i++){
        gsl_vector_set(alpha_gsl, i, alpha[i]);
    }

    /*
    printf("%f\n", bestBindingEnergy(alpha_gsl));
    return 0;
    */

    // Function to minimize
    gsl_multimin_function func;
    func.n = n;
    func.f = &bestBindingEnergy;

    // Set initial step sizes to 0.1
    gsl_vector * ss = gsl_vector_alloc(n);
    gsl_vector_set_all(ss, 0.1);

    // Environment variables for minimization
    const gsl_multimin_fminimizer_type * T = gsl_multimin_fminimizer_nmsimplex2;
    gsl_multimin_fminimizer * s = gsl_multimin_fminimizer_alloc(T, n);
    gsl_multimin_fminimizer_set(s, &func, alpha_gsl, ss);

    int iter = 0; // iteration counter
    int status;
    double size;
    do{
        iter++; // increment iteration counter
        status = gsl_multimin_fminimizer_iterate(s); // make the step
        if(status){ // check if anything went wrong
            break;
        }
        size = gsl_multimin_fminimizer_size(s); // update the size of the set
        status = gsl_multimin_test_size(size, 1e-8); // check if the size is sufficiently small
        if(status == GSL_SUCCESS){
            printf ("converged to minimum at\n");
        }
        printf("%4d f(", iter);
        for(int i = 0; i < n; i++){
                printf(" %.8f ", gsl_vector_get(s->x, i));
        }
        printf(") = %.8f   size = %.8f\n", s->fval, size);
    }while(status == GSL_CONTINUE && iter < 1000);

    return status;
}
