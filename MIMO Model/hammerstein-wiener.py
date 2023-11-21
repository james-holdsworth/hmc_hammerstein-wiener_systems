import platform
if platform.system()=='Darwin':
    import multiprocessing
    multiprocessing.set_start_method("fork")

# general imports
import pystan
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle


class Parameters:
    ...

class Data:
    ...

def lti(x, u, ω, A,B,C,D):
    y = C@x + D@u
    x = A@x + B@u + ω
    return x,y

def sat(u,upper,lower):
    u = upper if u > upper else u
    u = lower if u < lower else u
    return u

def dzone(u,left,right):
    y = 0
    y = u - left if u < left else y
    y = u - right if u > right else y
    return y

def MIMO_2x2_example(T, plot_me = True):

    par = Parameters()
    # system sizes
    par.n_u = 2
    par.n_y = 2
    par.n_x = 4

    # input nonlinearity truths
    par.sat_lower1 = -6.
    par.sat_upper1 = 6.
    par.dzone_left1 = -2.
    par.dzone_right1 = 2.
    par.alpha = np.array([par.sat_upper1,par.sat_lower1,par.dzone_left1,par.dzone_right1])

    # output nonlinearity truths
    par.dzone_left2 = -2.
    par.dzone_right2 = 2.
    par.sat_lower2 = -5
    par.sat_upper2 = 5
    par.beta = np.array([par.dzone_left2,par.dzone_right2,par.sat_upper2,par.sat_lower2])

    # linear system truths
    # Converted 4th order state space from Automatica paper, transfer function model via subspace + refinement
    par.A = np.array([[-0.024523620064597,   0.234584563723593,  -0.305707254662992,   0.018543780534269],
                    [ 0.030847977137114,  -0.373575676737083,   0.328847355994889,  -0.054782036427097],
                    [-0.152531048119485,   1.017691071266128,   0.280089201091929,   0.010327770978548],
                    [-0.675824092453352,   0.297333057752354,  -0.662923652782273,   0.888010095736634]], dtype = np.float64)

    par.B = np.array([[ 0.015096988944748,   0.002076099180373],
                    [ 0.003278444657333,  -0.007844677565818],
                    [ 0.001411430165043,  -0.015489965684947],
                    [ 0.001569567819373,  -0.013878203630782]], dtype = np.float64)

    par.C = np.array([[ -2.243189117364193, -26.193714897189544,  20.792628497591011, -33.844112878994345],
                    [-18.898901075772695,  -0.345713205812388, -21.907191710967155,  30.035215453590148]], dtype = np.float64)

    par.D = np.array([[ 1.100006390153607,   0.000011904983774],
                    [-0.860024547340833,  -0.000045732186252]], dtype = np.float64)
    
    # 2nd order PEM fit to the 4th order model from Automatica paper, via subspace + pem refinement
    par.A2 = np.array([[0.541,  0.05651],
                       [3.634,   0.3706]], dtype = np.float64)
    
    par.B2 = np.array([[-0.01203,  0.004875],
                       [0.002331,  -0.01265]], dtype = np.float64)
    
    par.C2 = np.array([[15.85,  -10.08],
                       [17.37,   4.432]], dtype = np.float64)
    
    par.D2 = np.array([[1.108,  -0.0006284],
                       [-0.8678,  -0.0006972]], dtype = np.float64)

    par.x_0 = np.array([[0.],[0.],[0.],[0.]]) # initial state assumed zero

    # noise covariances
    par.q = 0.1
    par.sq = np.sqrt(par.q)
    par.r = 0.1
    par.sr = np.sqrt(par.r)
    par.Q = par.q*np.eye(par.n_x) # ω (process noise) covariance
    par.R = par.r*np.eye(par.n_y) # e (output error) covariance

    u = np.zeros((par.n_u,T))
    w = np.zeros((par.n_u,T))
    x = np.zeros((par.n_x,T+1))
    s = np.zeros((par.n_y,T))

    # noise corrupted output 
    z = np.zeros((par.n_y,T))

    # noises
    ω = par.sq*np.random.standard_normal((par.n_y,T)) # process noise 
    e = par.sr*np.random.standard_normal((par.n_y,T)) # output noise 

    i = np.zeros((par.n_y,T))
    y = np.zeros((par.n_y,T))

    # create some inputs that are random
    u[0,:] = np.random.normal(0,7, T)
    u[1,:] = np.random.normal(0,7, T)

    # sim loop
    x[:,[0]] = par.x_0
    for k in range(T):
        w[0,k] = sat(u[0,k],par.sat_upper1,par.sat_lower1)
        w[1,k] = dzone(u[1,k],par.dzone_left1,par.dzone_right1)
        x[:,k+1],s[:,k] = lti(x[:,k],w[:,k], np.zeros(par.n_x), par.A,par.B,par.C,par.D)
        z[:,k] = s[:,k] + ω[:,k]
        i[0,k] = dzone(z[0,k],par.dzone_left2,par.dzone_right2)
        i[1,k] = sat(z[1,k],par.sat_upper2,par.sat_lower2)
        y[:,k] = i[:,k] + e[:,k]
    
    sim_data = Data()
    sim_data.u = u
    sim_data.w = w
    sim_data.x = x
    sim_data.s = s
    sim_data.z = z
    sim_data.i = i
    sim_data.y = y

    sim_data.T = T

    if plot_me:
        plt.figure()
        plt.plot(y[0,:])
        plt.plot(z[0,:])

        plt.figure()
        plt.plot(y[1,:])
        plt.plot(z[1,:])

        plt.figure()
        plt.plot(u[0,:])
        plt.plot(w[0,:])

        plt.figure()
        plt.plot(u[1,:])
        plt.plot(w[1,:])
        plt.show()
    return par, sim_data


def chain_starting_location(par,sim_data,N):
    chain = Parameters()
    chain.n_x = par.n_x
    chain.T = N
    # perturbed values acting as an initial 'estimate'
    chain.sat_lower1_start = par.sat_lower1 + np.random.uniform(-0.1,0.1)
    chain.sat_upper1_start = par.sat_upper1 + np.random.uniform(-0.1,0.1)
    chain.dzone_left1_start= par.dzone_left1 + np.random.uniform(-0.1,0.1)
    chain.dzone_right1_start = par.dzone_right1 + np.random.uniform(-0.1,0.1)
    chain.dzone_left2_start = par.dzone_left2 + np.random.uniform(-0.1,0.1)
    chain.dzone_right2_start = par.dzone_right2 + np.random.uniform(-0.1,0.1)
    chain.sat_lower2_start = par.sat_lower2 + np.random.uniform(-0.1,0.1)
    chain.sat_upper2_start = par.sat_upper2 + np.random.uniform(-0.1,0.1)

    chain.alpha_start = np.array([chain.sat_upper1_start,chain.sat_lower1_start,chain.dzone_left1_start,chain.dzone_right1_start])
    chain.beta_start = np.array([chain.dzone_left2_start,chain.dzone_right2_start,chain.sat_upper2_start,chain.sat_lower2_start])

    # 1% uniform error  --> starting at an 'estimate' that is close to the truth
    chain.A_start = par.A + np.multiply(0.01*np.abs(par.A),np.random.uniform(-1.,1.,(chain.n_x,chain.n_x)))
    chain.B_start = par.B + np.multiply(0.01*np.abs(par.B),np.random.uniform(-1.,1.,(chain.n_x,par.n_u)))                
    chain.C_start = par.C + np.multiply(0.01*np.abs(par.C),np.random.uniform(-1.,1.,(par.n_y,chain.n_x)))  
    chain.D_start = par.D + np.multiply(0.01*np.abs(par.D),np.random.uniform(-1.,1.,(par.n_y,par.n_u)))  


    x_sim = np.zeros((par.n_x,T+1))
    x_sim[:,[0]] = par.x_0 # consistent 'pure' state 
    chain.x_0 = par.x_0
    chain.w_sim = np.zeros((par.n_u,N))
    chain.s_sim = np.zeros((par.n_y,N))

    for k in range(N):
        chain.w_sim[0,k] = sat(sim_data.u[0,k],chain.sat_upper1_start,chain.sat_lower1_start)
        chain.w_sim[1,k] = dzone(sim_data.u[1,k],chain.dzone_left1_start,chain.dzone_right1_start)
        x_sim[:,k+1],chain.s_sim[:,k] = lti(x_sim[:,k],chain.w_sim[:,k], np.zeros(chain.n_x),chain.A_start,chain.B_start,chain.C_start,chain.D_start)

    chain.ln_sq_start = np.log(par.sq)*np.ones(par.n_y)
    chain.ln_sr_start = np.log(par.sr)*np.ones(par.n_y)
    return chain

def chain_starting_location_o2(par,sim_data,N):
    chain = Parameters()
    chain.n_x = 2
    chain.T = N
    # perturbed values acting as an initial 'estimate'
    chain.sat_lower1_start = par.sat_lower1 + np.random.uniform(-0.1,0.1)
    chain.sat_upper1_start = par.sat_upper1 + np.random.uniform(-0.1,0.1)
    chain.dzone_left1_start= par.dzone_left1 + np.random.uniform(-0.1,0.1)
    chain.dzone_right1_start = par.dzone_right1 + np.random.uniform(-0.1,0.1)
    chain.dzone_left2_start = par.dzone_left2 + np.random.uniform(-0.1,0.1)
    chain.dzone_right2_start = par.dzone_right2 + np.random.uniform(-0.1,0.1)
    chain.sat_lower2_start = par.sat_lower2 + np.random.uniform(-0.1,0.1)
    chain.sat_upper2_start = par.sat_upper2 + np.random.uniform(-0.1,0.1)

    chain.alpha_start = np.array([chain.sat_upper1_start,chain.sat_lower1_start,chain.dzone_left1_start,chain.dzone_right1_start])
    chain.beta_start = np.array([chain.dzone_left2_start,chain.dzone_right2_start,chain.sat_upper2_start,chain.sat_lower2_start])

    # 1% uniform error  --> starting at an 'estimate' that is close to the truth
    chain.A_start = par.A2 + np.multiply(0.01*np.abs(par.A2),np.random.uniform(-1.,1.,(chain.n_x,chain.n_x)))
    chain.B_start = par.B2 + np.multiply(0.01*np.abs(par.B2),np.random.uniform(-1.,1.,(chain.n_x,par.n_u)))                
    chain.C_start = par.C2 + np.multiply(0.01*np.abs(par.C2),np.random.uniform(-1.,1.,(par.n_y,chain.n_x)))  
    chain.D_start = par.D2 + np.multiply(0.01*np.abs(par.D2),np.random.uniform(-1.,1.,(par.n_y,par.n_u)))  


    x_sim = np.zeros((chain.n_x,T+1))
    chain.x_0 = np.zeros((chain.n_x,1))
    x_sim[:,[0]] = chain.x_0 # consistent 'pure' state 
    chain.w_sim = np.zeros((par.n_u,N))
    chain.s_sim = np.zeros((par.n_y,N))
    
    for k in range(N):
        chain.w_sim[0,k] = sat(sim_data.u[0,k],chain.sat_upper1_start,chain.sat_lower1_start)
        chain.w_sim[1,k] = dzone(sim_data.u[1,k],chain.dzone_left1_start,chain.dzone_right1_start)
        x_sim[:,k+1],chain.s_sim[:,k] = lti(x_sim[:,k],chain.w_sim[:,k], np.zeros(chain.n_x),chain.A_start,chain.B_start,chain.C_start,chain.D_start)

    chain.ln_sq_start = np.log(par.sq)*np.ones(par.n_y)
    chain.ln_sr_start = np.log(par.sr)*np.ones(par.n_y)
    return chain

def stan_chain_init(chain):
    return {'A': chain.A_start, # start values that have a random uniform up to 5% error on every element
            'B': chain.B_start, 
            'C': chain.C_start, 
            'D': chain.D_start, 
            'alpha': chain.alpha_start, # randomly purturbed values 
            'beta': chain.beta_start,
            'z': chain.s_sim[:,0:chain.T], # look at initialising all parameters in a `consistent' max likelihood solution
            'x0_p': chain.x_0.squeeze(),
            'ln_sq': chain.ln_sq_start,
            'ln_sr': chain.ln_sr_start }

def stan_data_init(chain, par, sim_data, N):
    return {'N': N,
            'n_u':par.n_u,
            'n_y':par.n_y,
            'n_x':chain.n_x,
            'x0': chain.x_0.squeeze(),
            'Q0': 0.1*np.eye(chain.n_x),
            'y': sim_data.y[:,0:N],
            'u': sim_data.u[:,0:N] }


## PROGRAM
if __name__ == "__main__":
    folder = 'MIMO Model/run 2/' # output folder for the runs

    T = 320 # data generation length

    par, sim_data = MIMO_2x2_example(T)

    Ns = [10, 20, 40, 50, 80, 100, 160, 200, 320] # use the first N data points
    for N in Ns:
        chain_start_o2 = chain_starting_location_o2(par, sim_data, N)
        stan_init = [stan_chain_init(chain_start_o2)]#, init_function(1), init_function(2), init_function(3)]
        stan_data = stan_data_init(chain_start_o2, par, sim_data, N)

        #sampling parameters
        M = 2000
        wup = 1000
        chains = 1
        iter = wup + int(M/chains)
        model_name = 'nd_hw_latent_z_diag_noise'
        path = 'MIMO Model/stan/'
        if Path(path+model_name+'.pkl').is_file():
            model = pickle.load(open(path+model_name+'.pkl', 'rb'))
        else:
            model = pystan.StanModel(file=path+model_name+'.stan')
            with open(path+model_name+'.pkl', 'wb') as file:
                pickle.dump(model, file)

        # with suppress_stdout_stderr():
        fit = model.sampling(data=stan_data, warmup=wup, iter=iter, init=stan_init,chains=chains, control=dict(metric = "dense_e")) # force dense mass matrix
        traces = fit.extract()

        # pickle run info
        run =  folder + 'run_latent_z_o2_results_T'+str(N)+'_W' + str(wup)+'_M'+str(M)

        with open(run+'_traces.pkl','wb') as file:
            pickle.dump(traces,file)
        with open(run+'_data.pkl','wb') as file:
            pickle.dump(sim_data,file)
        with open(run+'_par.pkl','wb') as file:
            pickle.dump(par,file)

    for N in Ns:
        chain_start = chain_starting_location(par, sim_data, N)
        stan_init = [stan_chain_init(chain_start)]#, init_function(1), init_function(2), init_function(3)]
        stan_data = stan_data_init(chain_start, par, sim_data, N)

        #sampling parameters
        M = 2000
        wup = 1000
        chains = 1
        iter = wup + int(M/chains)
        model_name = 'nd_hw_latent_z_diag_noise'
        path = 'MIMO Model/stan/'
        if Path(path+model_name+'.pkl').is_file():
            model = pickle.load(open(path+model_name+'.pkl', 'rb'))
        else:
            model = pystan.StanModel(file=path+model_name+'.stan')
            with open(path+model_name+'.pkl', 'wb') as file:
                pickle.dump(model, file)

        # with suppress_stdout_stderr():
        fit = model.sampling(data=stan_data, warmup=wup, iter=iter, init=stan_init,chains=chains, control=dict(metric = "dense_e")) # force dense mass matrix
        traces = fit.extract()

        # pickle run info
        run =  folder + 'run_latent_z_results_T'+str(N)+'_W' + str(wup)+'_M'+str(M)

        with open(run+'_traces.pkl','wb') as file:
            pickle.dump(traces,file)
        with open(run+'_data.pkl','wb') as file:
            pickle.dump(sim_data,file)
        with open(run+'_par.pkl','wb') as file:
            pickle.dump(par,file)

    y_hat = traces['y_hat_out']
    PT = y_hat.shape[2]
    ts = np.arange(y_hat.shape[2])
    err1 = np.zeros((y_hat.shape[0],y_hat.shape[2]))
    err2 = np.zeros((y_hat.shape[0],y_hat.shape[2]))
    for ii in ts:
        err1[:,ii] = y_hat[:,0,ii] - sim_data.y[0,ii]
        err2[:,ii] = y_hat[:,1,ii] - sim_data.y[1,ii]

    fig = plt.figure(figsize=(6.4,5),dpi=300)
    plt.plot(ts,sim_data.y[0,0:PT] + err1.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1,label='Mean of error')
    plt.fill_between(ts,sim_data.y[0,0:PT] +  np.percentile(err1,97.5,axis=0),sim_data.y[0,0:PT] + np.percentile(err1,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% interval')
    plt.plot(ts,sim_data.y[0,0:PT], color=u'#1f0000',linestyle='--',linewidth = 1,label='True value')
    # plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,len(ts)])
    plt.ylabel('Prediction error dist., output 1')
    plt.show()

    fig = plt.figure(figsize=(6.4,5),dpi=300)
    plt.plot(ts,sim_data.y[1,0:PT] + err2.mean(axis = 0), color=u'#1f77b4',linestyle='--',linewidth = 1,label='Mean of error')
    plt.fill_between(ts,sim_data.y[1,0:PT] +  np.percentile(err2,97.5,axis=0),sim_data.y[1,0:PT] + np.percentile(err2,2.5,axis=0),color=u'#1f77b4',alpha=0.15,label='95% interval')
    plt.plot(ts,sim_data.y[1,0:PT], color=u'#1f0000',linestyle='--',linewidth = 1,label='True value')
    # plt.ticklabel_format(style='sci',axis='y',scilimits=(0,0))
    plt.xlim([0,len(ts)])
    plt.ylabel('Prediction error dist., output 2')
    plt.show()

    1+1

    # y_pred = np.zeros((1,M))
    # w_pred
    # for k in range(T):
    #     w_pred[k] = hinges(u[k],par)
    #     x[k+1],s[k] = lti(x[k],w[k],par)
    #     ξ[k+1],ν[k],μ[k] = corr_process(ξ[k],v[k],e[k],par)
    #     z[k] = s[k] + ν[k]
    #     i[k] = sat(z[k],par)
    #     y[k] = i[k] + μ[k]

    # dictionary = {
    #     "Q":q_samps,
    #     "a":a_samps,
    #     # "c":c_samps
    # }




