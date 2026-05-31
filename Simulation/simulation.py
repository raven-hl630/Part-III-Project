#Import packages
import numpy as np
import matplotlib.pyplot as plt
import iminuit
import multiprocessing
import json


#Define configuration
class configuration():
    def __init__(self):
        
        #Define mixing and decay parameters
        self.modulus_q_p = 1.0010
        self.x = 0.7697
        self.y = - 0.0005
        self.C_f = 0.002
        self.S_f = 0.711
        self.D_f = - np.sqrt(1 - self.C_f**2 - self.S_f**2)
        
        #Define other decay parameters
        self.m_X_f_mean = 5279.66
        self.m_X_f_width = 4.0
        self.m_Xb_f_mean = 5279.66
        self.m_Xb_f_width = 4.0
        self.tau_bkg = 1
        self.m_bkg_mean = 5279.66
        self.m_bkg_width = 400
        
        #Define detector, reconstruction, and selection parameters
        #If f_sig=1, sample has no background events, recommend defining bkg parameters as np.nan
        self.N_sig_expected = 27 * 10**6        #N_sig is a run parameter
        self.f_sig = 0.351
        self.omega = 0.25                       #omega is a run parameter
        self.eps_X_f = 0.470
        self.eps_Xb_f = 0.470
        
        #Define run parameters
        self.run_number = 1000
        self.N_sig = 10**5
        self.omega_list_cfg = np.linspace(0, 0.4, 17).tolist()


def function(omega_pool):
    
    #Transfer configuration from global to local  
    cfg = configuration()
    
    #Retrieve mixing and decay parameters
    modulus_q_p = cfg.modulus_q_p
    x = cfg.x
    y = cfg.y
    C_f = cfg.C_f
    S_f = cfg.S_f
    D_f = cfg.D_f
    
    #Retrieve other decay parameters
    m_X_f_mean = cfg.m_X_f_mean
    m_X_f_width = cfg.m_X_f_width
    m_Xb_f_mean = cfg.m_Xb_f_mean
    m_Xb_f_width = cfg.m_Xb_f_width
    tau_bkg = cfg.tau_bkg
    m_bkg_mean = cfg.m_bkg_mean
    m_bkg_width = cfg.m_bkg_width
    
    #Retrieve detector, reconstruction, and selection parameters
    N_sig_expected = cfg.N_sig_expected                     #N_sig is a run parameter
    f_sig = cfg.f_sig
    omega = cfg.omega                                       #omega is a run parameter
    eps_X_f = cfg.eps_X_f
    eps_Xb_f = cfg.eps_Xb_f
    
    #Retrieve run parameters
    run_number = cfg.run_number
    N_sig = cfg.N_sig                                       #N_sig is chosen for sample size
    omega_array_cfg = np.array(cfg.omega_list_cfg)          #omega is multiprocessed
    omega = omega_pool
    
    
    #Define PDFs for decay time and invariant mass
    
    #Define PDF of decay time (Gamma*t for signal PDFs)
    def PDF_t_X_f(t, x, y, C_f, D_f, S_f):
        factor_X_f = 1/(1-y**2) * (1+y*D_f)
        factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
        PDF_t_X_f = 1/(factor_X_f + factor_Y_f) * np.exp(-t) * (np.cosh(y*t)+D_f*np.sinh(y*t)+C_f*np.cos(x*t)-S_f*np.sin(x*t))
        return np.where(t>=0, PDF_t_X_f, 0)
    
    def PDF_t_Xb_f(t, x, y, C_f, D_f, S_f):
        factor_X_f = 1/(1-y**2) * (1+y*D_f)
        factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
        PDF_t_Xb_f = 1/(factor_X_f - factor_Y_f) * np.exp(-t) * (np.cosh(y*t)+D_f*np.sinh(y*t)-C_f*np.cos(x*t)+S_f*np.sin(x*t))
        return np.where(t>=0, PDF_t_Xb_f, 0)
    
    def PDF_t_bkg(t, tau_bkg):
        PDF_t_bkg = 1/tau_bkg * np.exp(-t/tau_bkg)
        return np.where(t>=0, PDF_t_bkg, 0)
    
    #Define PDF of invariant mass m
    def PDF_m_X_f(m, m_X_f_mean, m_X_f_width):
        PDF_m_X_f = 1/(np.sqrt(2*np.pi)*m_X_f_width) * np.exp(-((m-m_X_f_mean)**2)/(2*m_X_f_width**2))
        return np.where(m>=0, PDF_m_X_f, 0)
    
    def PDF_m_Xb_f(m, m_Xb_f_mean, m_Xb_f_width):
        PDF_m_Xb_f = 1/(np.sqrt(2*np.pi)*m_Xb_f_width) * np.exp(-((m-m_Xb_f_mean)**2)/(2*m_Xb_f_width**2))
        return np.where(m>=0, PDF_m_Xb_f, 0)
    
    def PDF_m_bkg(m, m_bkg_mean, m_bkg_width):
        PDF_m_bkg = np.zeros(len(m))
        PDF_m_bkg[(m>=(m_bkg_mean-0.5*m_bkg_width)) & (m<=(m_bkg_mean+0.5*m_bkg_width))] = 1/m_bkg_width
        return np.where(m>=0, PDF_m_bkg, 0)
    
    
    #Define CDFs for use in event generation
    def CDF_t_X_f(t, x, y, C_f, D_f, S_f):
        factor_X_f = 1/(1-y**2) * (1+y*D_f)
        factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
        CDF_t_X_f = - 1/(factor_X_f + factor_Y_f) * ( np.exp(-t) * (1/(1-y**2)*((1+y*D_f)*np.cosh(y*t)+(y+D_f)*np.sinh(y*t)) + 1/(1+x**2)*((C_f-x*S_f)*np.cos(x*t)+(-x*C_f-S_f)*np.sin(x*t))) - (1/(1-y**2)*(1+y*D_f) + 1/(1+x**2)*(C_f-x*S_f)) )
        return np.where(t>=0, CDF_t_X_f, 0)
    
    def CDF_t_Xb_f(t, x, y, C_f, D_f, S_f):
        factor_X_f = 1/(1-y**2) * (1+y*D_f)
        factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
        CDF_t_Xb_f = - 1/(factor_X_f - factor_Y_f) * ( np.exp(-t) * (1/(1-y**2)*((1+y*D_f)*np.cosh(y*t)+(y+D_f)*np.sinh(y*t)) + 1/(1+x**2)*((-C_f+x*S_f)*np.cos(x*t)+(x*C_f+S_f)*np.sin(x*t))) - (1/(1-y**2)*(1+y*D_f) + 1/(1+x**2)*(-C_f+x*S_f)) )
        return np.where(t>=0, CDF_t_Xb_f, 0)
    
    
    #Define the sampling function that returns samples of decay time, invariant mass, and tag
    def sample_generation_simple():
        
        #Compute the ratio of branching fractions
        factor_X_f = 1/(1-y**2) * (1+y*D_f)
        factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
        BF_ratio_Xb_f_X_f = 1/(modulus_q_p**2) * (factor_X_f - factor_Y_f)/(factor_X_f + factor_Y_f)
        #Compute numbers of true X->f and Xb->f signal events
        N_X_f = np.random.poisson(1/(1+BF_ratio_Xb_f_X_f) * N_sig)
        N_Xb_f = np.random.poisson(BF_ratio_Xb_f_X_f/(1+BF_ratio_Xb_f_X_f) * N_sig)
        #Compute the number of background events in the accepted sample
        N_bkg = np.random.poisson((1-f_sig)/f_sig * (eps_X_f*1/(1+BF_ratio_Xb_f_X_f) + eps_Xb_f*BF_ratio_Xb_f_X_f/(1+BF_ratio_Xb_f_X_f)) * N_sig)
        
        #Generate decay time data for signal and background events
        #Using a sample from a uniform distribution between 0 and 1 as the argument of the inverse CDF gives a sample from the corresponding probability distribution
        #For signal events
        u_t = np.random.uniform(0, 1, N_X_f)
        v_t = np.random.uniform(0, 1, N_Xb_f)
        t_range = np.linspace(0, 100, 100000)
        sample_t_X_f = np.interp(u_t, CDF_t_X_f(t_range, x, y, C_f, D_f, S_f), t_range)
        sample_t_Xb_f = np.interp(v_t, CDF_t_Xb_f(t_range, x, y, C_f, D_f, S_f), t_range)
        #For background events
        if N_bkg == 0:
            sample_t_bkg = np.array([])
        if N_bkg > 0:
            sample_t_bkg = np.random.exponential(tau_bkg, N_bkg)
        
        #Generate invariant mass data for signal and background events
        sample_m_X_f = np.random.normal(m_X_f_mean, m_X_f_width, N_X_f)
        sample_m_Xb_f = np.random.normal(m_Xb_f_mean, m_Xb_f_width, N_Xb_f)
        if N_bkg == 0:
            sample_m_bkg = np.array([])
        if N_bkg > 0:
            sample_m_bkg = np.random.uniform(m_bkg_mean-0.5*m_bkg_width, m_bkg_mean+0.5*m_bkg_width, N_bkg)
        
        #Incorporate acceptance of true signal events
        #Acceptance does not apply to background events, as they are only defined in the analysis sample and are not well defined prior to selection
        #Assume an X/Xb->f/bkg event occuring at decay time t has a probability eps_X_f/Xb_f/bkg(t) to be accepted
        u_acc = np.random.uniform(0, 1, N_X_f)
        v_acc = np.random.uniform(0, 1, N_Xb_f)
        #If u/v/w <= eps(t), the event is accepted; if u/v/w > eps(t), the event is not accepted (if eps=1, all events are accepted)
        #Compute the accepted event indices
        acceptance_X_f = (u_acc <= eps_X_f)
        acceptance_Xb_f = (v_acc <= eps_Xb_f)
        #Retain only the accepted signal events
        sample_acc_t_X_f = sample_t_X_f[acceptance_X_f]
        sample_acc_t_Xb_f = sample_t_Xb_f[acceptance_Xb_f]
        sample_acc_m_X_f = sample_m_X_f[acceptance_X_f]
        sample_acc_m_Xb_f = sample_m_Xb_f[acceptance_Xb_f]
        N_acc_X_f = np.sum(acceptance_X_f)
        N_acc_Xb_f = np.sum(acceptance_Xb_f)
        
        #Incorporate mistagging
        #Assume an X/Xb->f signal event has a probability (1-omega) to be correctly tagged and a probability omega to be mistagged
        #Assuming no biase in the tag assignment of background events (random tag assignment), mistagging does not affect the tagging statistics of background events
        u_tag = np.random.uniform(0, 1, N_acc_X_f)
        v_tag = np.random.uniform(0, 1, N_acc_Xb_f)
        #If u/v < omega, the tag is incorrect; if u/v >= omega, the tag is correct (if omega=0, all tags are correctly assigned)
        #Tagging label for (X, Xbar) is (1, -1)
        #Order decay time events into two samples based on the generated tags
        sample_acc_tag_t_X_f = np.concatenate((sample_acc_t_X_f[u_tag >= omega], sample_acc_t_Xb_f[v_tag < omega]))
        sample_acc_tag_t_Xb_f = np.concatenate((sample_acc_t_Xb_f[v_tag >= omega], sample_acc_t_X_f[u_tag < omega]))
        sample_acc_tag_m_X_f = np.concatenate((sample_acc_m_X_f[u_tag >= omega], sample_acc_m_Xb_f[v_tag < omega]))
        sample_acc_tag_m_Xb_f = np.concatenate((sample_acc_m_Xb_f[v_tag >= omega], sample_acc_m_X_f[u_tag < omega]))
        N_acc_tag_X_f = np.sum(u_tag >= omega) + np.sum(v_tag < omega)
        N_acc_tag_Xb_f = np.sum(v_tag >= omega) + np.sum(u_tag < omega)
        
        #Construct observed sample: decay time, invariant mass, tag
        sample_t = np.concatenate((sample_acc_tag_t_X_f, sample_acc_tag_t_Xb_f, sample_t_bkg))
        sample_m = np.concatenate((sample_acc_tag_m_X_f, sample_acc_tag_m_Xb_f, sample_m_bkg))
        sample_q = np.concatenate((np.ones(N_acc_tag_X_f), -np.ones(N_acc_tag_Xb_f), np.random.choice([1, -1], size=N_bkg, p=[0.5, 0.5])))
        
        return sample_t, sample_m, sample_q
    
    
    #Define the effective PDF of a dataset (t, m, q)
    def PDF_simple(t, m, q, C_f, D_f, S_f, f_sig):
        #Define Kronecker delta for tag selection
        delta_q_X_f = np.where(q == 1, 1, 0)
        delta_q_Xb_f = np.where(q == -1, 1, 0)
        #Compute ratio of branching fractions
        factor_X_f = 1/(1-y**2) * (1+y*D_f)
        factor_Y_f = 1/(1+x**2) * (C_f-x*S_f)
        BF_ratio_Xb_f_X_f = 1/(modulus_q_p**2) * (factor_X_f - factor_Y_f)/(factor_X_f + factor_Y_f)
        #Compute the probabilities of true signal decay modes P(X/Xb->f)
        P_X_f = f_sig/eps_X_f * 1/(1+BF_ratio_Xb_f_X_f)
        P_Xb_f = f_sig/eps_Xb_f * BF_ratio_Xb_f_X_f/(1+BF_ratio_Xb_f_X_f)
        #Compute the probability of background modes P(bkg) in the accepted sample
        P_bkg = 1 - f_sig
        #Compute conditional tagging probabilities given the true source P(q=X/Xb|X/Xb/bkg)
        #Define P_A_q_B as the probability of A being tagged as q=B
        P_X_f_q_X_f = 1 - omega
        P_X_f_q_Xb_f = omega
        P_Xb_f_q_X_f = omega
        P_Xb_f_q_Xb_f = 1 - omega
        P_bkg_q_X_f = 0.5
        P_bkg_q_Xb_f = 0.5
        #Compute sample PDFs of decay time and invariant mass given the true source
        P_t_X_f = PDF_t_X_f(t, x, y, C_f, D_f, S_f)
        P_t_Xb_f = PDF_t_Xb_f(t, x, y, C_f, D_f, S_f)
        P_m_X_f = PDF_m_X_f(m, m_X_f_mean, m_X_f_width)
        P_m_Xb_f = PDF_m_Xb_f(m, m_Xb_f_mean, m_Xb_f_width)
        P_t_bkg = PDF_t_bkg(t, tau_bkg)
        P_m_bkg = PDF_m_bkg(m, m_bkg_mean, m_bkg_width)
        #Compute the overall effective PDF
        PDF_X_f = P_t_X_f*P_m_X_f*P_X_f_q_X_f*eps_X_f*P_X_f + P_t_Xb_f*P_m_Xb_f*P_Xb_f_q_X_f*eps_Xb_f*P_Xb_f + P_t_bkg*P_m_bkg*P_bkg_q_X_f*P_bkg
        PDF_Xb_f = P_t_X_f*P_m_X_f*P_X_f_q_Xb_f*eps_X_f*P_X_f + P_t_Xb_f*P_m_Xb_f*P_Xb_f_q_Xb_f*eps_Xb_f*P_Xb_f + P_t_bkg*P_m_bkg*P_bkg_q_Xb_f*P_bkg
        PDF_simple = PDF_X_f*delta_q_X_f + PDF_Xb_f*delta_q_Xb_f
        return PDF_simple
    
    
    #Define the unbinned NLL cost function
    #Note that iminuit fits all the arguments of the cost function; sample data and parameters not meant to be fitted must be read globally
    def cost_function_simple(C_f, D_f, S_f, f_sig):
        #PDF_sample_simple = np.maximum(PDF_simple(sample_t, sample_m, sample_q, C_f, D_f, S_f, f_sig), 1e-300)
        PDF_sample_simple = PDF_simple(sample_t, sample_m, sample_q, C_f, D_f, S_f, f_sig)
        cost_function_simple = - np.sum(np.log(PDF_sample_simple))
        return cost_function_simple
    #Inform iminuit the cost function is in NLL form
    cost_function_simple.errordef = iminuit.Minuit.LIKELIHOOD
    
    
    #Initialise arrays for the storage of iminuit results
    failure_count = 0
    C_f_value_array_run = np.zeros(run_number)
    D_f_value_array_run = np.zeros(run_number)
    S_f_value_array_run = np.zeros(run_number)
    C_f_error_array_run = np.zeros(run_number)
    D_f_error_array_run = np.zeros(run_number)
    S_f_error_array_run = np.zeros(run_number)
    f_sig_value_array_run = np.zeros(run_number)
    f_sig_error_array_run = np.zeros(run_number)
    
    
    for run_index in range(run_number):
        
        while True:
            
            #Sample generation and fitting
            sample_t, sample_m, sample_q = sample_generation_simple()
            mi = iminuit.Minuit(cost_function_simple, C_f=C_f, D_f=D_f, S_f=S_f, f_sig=f_sig)
            mi.limits["C_f", "S_f"] = (-1, 1)
            mi.fixed["D_f"] = True
            mi.limits["f_sig"] = (0, 1)
            mi.migrad()
            mi.hesse()
            
            #Check validity, repeat if fit fails
            validity = mi.valid
            if validity == True:
                break
            if validity == False:
                failure_count += 1
        
        #Store iminuit results
        C_f_value_array_run[run_index] = mi.values["C_f"]
        D_f_value_array_run[run_index] = mi.values["D_f"]
        S_f_value_array_run[run_index] = mi.values["S_f"]
        C_f_error_array_run[run_index] = mi.errors["C_f"]
        D_f_error_array_run[run_index] = mi.errors["D_f"]
        S_f_error_array_run[run_index] = mi.errors["S_f"]
        f_sig_value_array_run[run_index] = mi.values["f_sig"]
        f_sig_error_array_run[run_index] = mi.errors["f_sig"]
    
    failure_rate = failure_count/(run_number + failure_count)
    result_array_param_run = np.array([C_f_value_array_run, D_f_value_array_run, S_f_value_array_run, C_f_error_array_run, D_f_error_array_run, S_f_error_array_run, f_sig_value_array_run, f_sig_error_array_run])
    
    return failure_rate, result_array_param_run


if __name__ == "__main__":
    
    #Define multiprocessed cfg parameter
    cfg = configuration()
    omega_array_cfg = np.array(cfg.omega_list_cfg)
    cfg_number = len(omega_array_cfg)
    
    #Set start method to "spawn" so the processes do not inherit the same seed for random number generation
    multiprocessing.set_start_method("spawn")
    #Perform fitting
    with multiprocessing.Pool(cfg_number) as pool:
        result_list_cfg_tuple = pool.map(function, omega_array_cfg)
        failure_rate_array_cfg, result_array_cfg_param_run = map(np.array, zip(*result_list_cfg_tuple))
    
    #Unpack results
    C_f_value_array_cfg_run = result_array_cfg_param_run[:, 0]
    D_f_value_array_cfg_run = result_array_cfg_param_run[:, 1]
    S_f_value_array_cfg_run = result_array_cfg_param_run[:, 2]
    C_f_error_array_cfg_run = result_array_cfg_param_run[:, 3]
    D_f_error_array_cfg_run = result_array_cfg_param_run[:, 4]
    S_f_error_array_cfg_run = result_array_cfg_param_run[:, 5]
    f_sig_value_array_cfg_run = result_array_cfg_param_run[:, 6]
    f_sig_error_array_cfg_run = result_array_cfg_param_run[:, 7]
    
    #Save results
    with open("run_0_configuration.json", "w") as f:
        json.dump(cfg.__dict__, f, indent=4)
    np.save("run_0_failure_rate_array_cfg.npy", failure_rate_array_cfg)
    np.save("run_0_C_f_value_array_cfg_run.npy", C_f_value_array_cfg_run)
    np.save("run_0_D_f_value_array_cfg_run.npy", D_f_value_array_cfg_run)
    np.save("run_0_S_f_value_array_cfg_run.npy", S_f_value_array_cfg_run)
    np.save("run_0_C_f_error_array_cfg_run.npy", C_f_error_array_cfg_run)
    np.save("run_0_D_f_error_array_cfg_run.npy", D_f_error_array_cfg_run)
    np.save("run_0_S_f_error_array_cfg_run.npy", S_f_error_array_cfg_run)
    np.save("run_0_f_sig_value_array_cfg_run.npy", f_sig_value_array_cfg_run)
    np.save("run_0_f_sig_error_array_cfg_run.npy", f_sig_error_array_cfg_run)
