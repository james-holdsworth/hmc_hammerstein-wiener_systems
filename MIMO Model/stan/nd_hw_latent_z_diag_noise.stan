functions {
    real saturation(real nu, real upper, real lower) {
        real y;
        if (nu > upper) {
            y = upper;
        } else if (nu < lower) {
           y = lower;
        } else {
            y = nu;
        }
        return y;
    }

    real deadzone(real nu, real left, real right) {
        real y;
        y = 0;
        if(nu > right) {
            y = nu - right;
        } else if (nu < left) {
            y = nu - left;
        }
       return y;
    }
}

data {
    int<lower=0> N; // length of discretised time window
    int<lower=0> n_u; // number of inputs
    int<lower=0> n_y; // number of outputs
    int<lower=0> n_x; // number of states of linear model
    vector[n_x] x0; // initial system state
    matrix[n_x,n_x] Q0; // covariance of initial state 
    matrix[n_u,N] u;
    matrix[n_y,N] y;
}

parameters {
    // memory-less non-linear block parameters
    vector[4] alpha;
    vector[4] beta;

    // linear block parameters
    vector[n_x] x0_p;
    matrix[n_x,n_x] A;
    matrix[n_x,n_u] B;
    matrix[n_y,n_x] C;
    matrix[n_y,n_u] D;

    //vector[n_y] z[N];
    matrix[n_y,N] z;
    
    // covariance parameters for the e output noise and v driving noise processes
    // vector<lower=1e-9>[n_x] sq; // diagonal elements of diagonal sQ matrix 
    // cholesky_factor_corr[n_y] Q_corr_chol; // n_y by n_y lower cholesky of correlation matrix, give uninformative prior (LKJ)
    vector[n_y] ln_sq; // log of scale vector, give flat (improper) prior
    // cholesky_factor_corr[n_y] R_corr_chol; // n_y by n_y lower cholesky of correlation matrix, give uninformative prior (LKJ)
    vector[n_y] ln_sr; // log of scale vector, give flat (improper) prior
}

transformed parameters {
    matrix[n_x,N+1] x;
    matrix[n_u,N] w;
    matrix[n_y,N] z_hat_mat;
    matrix[n_y,N] y_hat_mat;
    vector[n_y] y_hat[N];

    x[:,1] = x0_p;
    for (i in 1:N){
        w[1,i] = saturation(u[1,i],alpha[1],alpha[2]);
        w[2,i] = deadzone(u[2,i],alpha[3],alpha[4]);
        x[:,i+1] = A*x[:,i] + B*w[:,i];
    }
    z_hat_mat = C*x[:,1:N] + D*w; // do not update x first
    for (i in 1:N){
        y_hat_mat[1,i] = deadzone(z[1,i],beta[1],beta[2]);
        y_hat_mat[2,i] = saturation(z[2,i],beta[3],beta[4]);
    }
}

model {
    x0_p ~ multi_normal(x0,Q0);
    z[1,:] ~ normal(z_hat_mat[1,:],exp(ln_sq[1]));
    z[2,:] ~ normal(z_hat_mat[2,:],exp(ln_sq[2]));
    y[1,:] ~ normal(y_hat_mat[1,:],exp(ln_sr[1]));
    y[2,:] ~ normal(y_hat_mat[2,:],exp(ln_sr[2]));
    // regularising prior 
    // alpha ~ normal(0,1000);
    // beta ~ normal(0,1000);

}
generated quantities {
    matrix[n_y,N] y_hat_out;
    y_hat_out[1,:] = to_row_vector(normal_rng(y_hat_mat[1,:],exp(ln_sr[1])));
    y_hat_out[2,:] = to_row_vector(normal_rng(y_hat_mat[2,:],exp(ln_sr[2])));
}