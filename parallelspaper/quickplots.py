import numpy as np
import matplotlib.pyplot as plt
from parallelspaper.model_fitting import get_y, powerlaw_decay, exp_decay, pow_exp_decay

### plot model fit
def plot_model_fits(MI, MI_shuff, distances, results_power, results_exp, results_pow_exp, zoom=4, var_MI=0):
    fig, axs = plt.subplots(ncols=5, figsize=(5*zoom,1*zoom))
    ax = axs[0]
    ax.plot(distances, MI)
    ax.plot(distances, MI_shuff)

    
    ax = axs[1]
    ax.set_title(results_power.aic)
    fit = get_y(powerlaw_decay, results_power, distances)
    ax.plot(distances, 
            fit,
           color = 'k')

    ax = axs[2]
    ax.set_title(results_exp.aic)
    fit = get_y(exp_decay, results_exp, distances)
    ax.plot(distances, 
            fit,
           color = 'k')

    # concatenative 
    ax = axs[3]
    ax.set_title(results_pow_exp.aic)
    fit = get_y(pow_exp_decay, results_pow_exp, distances)
    ax.plot(distances, 
            fit,
           color = 'k')
    fit = get_y(exp_decay, results_pow_exp, distances)
    ax.plot(distances, fit,color = 'k')
    fit = get_y(powerlaw_decay, results_pow_exp, distances)
    ax.plot(distances, fit,color = 'k')
    
    
    ax = axs[4]
    ax.set_title('powerlaw component')
    fit = get_y(exp_decay, results_pow_exp, distances)
    ax.scatter(distances, MI-MI_shuff-fit)
    fit = get_y(powerlaw_decay, results_pow_exp, distances) 
    ax.plot(distances, 
            fit- results_pow_exp.params['intercept'].value,
           color = 'k')
    
    ax.set_xscale( "log" , basex=10)
    ax.set_yscale( "log" , basey=10)
    
    sig = MI-MI_shuff
    sig_lims = np.log([np.min(sig[sig>0]), np.nanmax(sig)])
    #print(sig_lims)
    sig_lims = [sig_lims[0] - (sig_lims[1]-sig_lims[0])/10,
                    sig_lims[1] + (sig_lims[1]-sig_lims[0])/10]
    
    for ax in axs[1:-1]:
        
        ax.set_ylim(np.exp(sig_lims))
        ax.scatter(distances, MI-MI_shuff)
        ax.fill_between(distances, MI-MI_shuff- var_MI, MI-MI_shuff+ var_MI, alpha=0.5, color='#3F5866')
        ax.set_xscale( "log" , basex=10)
        ax.set_yscale( "log" , basey=10)
        ax.set_xlim([distances[0], distances[-1]])
    plt.show()