// gcc -o pwaves main.c -lm -lgsl -lgslcblas
/*
 *
 * I calculate the best expansion on a set of cartesianly antisymmetrized pwaves (gaussians multiplied by a cartesian direction variable) with given widhts alpha.
 * The minimum eigenvalue is returned. The procedure of minimization is carried out on the alpha parameters through a Nelder-Mead method.
 * Such to put the algorithm in a more generalized fashion I can choose both the number of cartesian directions I want to investigate (dir) and the number of pwaves to use (xi).
 * This last is the same for any direction (symmetry arguments to be advanced). For example if dir=2 and xi=3 it means that I'm going to use 3+3 pwaves over 2 directions (say x and y)
 *
 *
 */
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#include <gsl/gsl_matrix.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_multimin.h>


double pi(){
    return 4*atan(1);
}
int xi[3]; // numero di p-waves per direzione (x,y,z)
int dir;// numero di direzioni lungo cui prendiamo le p waves
int start=0;
int end=0;

double bestBindingEnergy(const gsl_vector * alpha){
    size_t n = alpha->size;
    
    double aa[n];
    for (int i =0; i<n; i++){
        aa[i]=gsl_vector_get(alpha,i);
    }
    
    gsl_matrix * S = gsl_matrix_alloc(n,n);
    gsl_matrix_set_zero(S);
    
    /*I will introduce non-zero values only in the correct entrances: overlap only over same direction p-waves
     */
    start=0;
    end=0;
    for(int pj = 0; pj < dir; pj++){
        
        start=end;
        end+=xi[pj];
        for(int i = start; i < end; i++){
            for(int j=i; j < end; j++){
                gsl_matrix_set(S, i, j, 0.5*pow(pi(),1.5)/pow(aa[i]+aa[j], 2.5));
                gsl_matrix_set(S, j, i, gsl_matrix_get(S,i,j));
            }
        }
    }
    
    gsl_matrix * H = gsl_matrix_alloc(n,n);
    gsl_matrix_set_zero(H);
    
    /*I will introduce non-zero values only in the correct entrances: interaction only over same direction p-waves
     */
    start=0;
    end=0;
    for(int pj = 0; pj < dir; pj++){
        start=end;
        end+=xi[pj];
        for(int i = start; i < end; i++){
            for(int j=i; j < end; j++){
                gsl_matrix_set(H, i, j, 2.5*pi()*sqrt( pi() )*aa[i]*aa[j]/(pow(aa[i]+aa[j],3.5))-2*pi()/(3*pow(aa[i]+aa[j],2)));
                gsl_matrix_set(H, j, i, gsl_matrix_get(H,i,j));
            }
        }
    }
    
    gsl_vector * eval = gsl_vector_alloc(n);
    gsl_matrix * evec = gsl_matrix_alloc(n,n);
    gsl_eigen_gensymmv_workspace * w =  gsl_eigen_gensymmv_alloc(n);
    gsl_eigen_gensymmv(H, S, eval, evec, w);
    
    return gsl_vector_min(eval);
}


int main(){
    
    int dim=0; // dimension of vector of parameter is Sum((directions in cartesian space I use)*(number of p-waves per direction)) // obviously ;p
    printf("Insert number of directions you want to use (1,2 or 3)\n");
    scanf("%d",&dir); // number of directions I want to use
    for(int i=0;i<dir;i++)
    {
        printf("Insert number of wavefuntions you want to use for per projection %d: \n", i);
        scanf("%d",&xi[i]); // number of p-waves per direction I want to use
        dim+=xi[i];
    }
    
    
    
    double alpha[dim];
    
    printf("Insert values of parameters of the gaussian basis \n");
    for(int i=0;i<dim; i++){
        scanf("%lf",&alpha[i]);
    }
    
    size_t n = sizeof(alpha)/sizeof(alpha[0]);
    printf("Dimension N %d: \n", n);
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
    gsl_vector_set_all(ss, 0.0001);
    
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
        status = gsl_multimin_test_size(size, 1e-10); // check if the size is sufficiently small
        if(status == GSL_SUCCESS){
            printf ("converged to minimum at\n");
        }
        
        printf("%4d f(", iter);
        for(int i = 0; i < n; i++){
            printf(" %f ", gsl_vector_get(s->x, i));
        }
        printf(") = %f   size = %f\n", s->fval, size);
    }while(status == GSL_CONTINUE && iter < 1000);
    
    gsl_vector_free(alpha_gsl);
    gsl_vector_free(ss);
    gsl_multimin_fminimizer_free (s);
    
    return status;
}
