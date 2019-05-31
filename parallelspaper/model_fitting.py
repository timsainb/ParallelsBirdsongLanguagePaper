import numpy as np
import lmfit
from scipy.stats.distributions import chi2

# model fit quality
def residuals(y_true, y_model, x, logscaled=False):
    if logscaled:
        return np.abs(np.log(y_true) - np.log(y_model)) * (1 / (np.log(1 + x)))
    else:
        return np.abs(y_true - y_model)


def RSS(y_true, y_model, x, logscaled=False):
    return np.sum(residuals(y_true, y_model, x, logscaled=logscaled) ** 2)


def AIC(N_data, N_params, y_true, y_model, x, logscaled=False):
    return (
        N_data * np.log(RSS(y_true, y_model, x, logscaled=logscaled) / N_data)
        + 2 * N_params
    )


def log_likelihood(N_data, y_true, y_model, x, logscaled=False):
    return -(N_data / 2) * np.log(RSS(y_true, y_model, x, logscaled) / N_data)


def AICc(N_data, N_params, y_true, y_model, x, logscaled=False):
    return AIC(N_data, N_params, y_true, y_model, x, logscaled=logscaled) + (
        2 * N_params * (N_params + 1)
    ) / (N_data - N_params - 1)


def delta_AIC(AICs):
    return AICs - np.min(AICs)


def relative_likelihood(delta_AIC):
    return np.exp(-0.5 * delta_AIC)


def Prob_model_Given_data_and_models(model_relative_likelihoods):
    """ probability of the model given data and the other models
    """
    return model_relative_likelihoods / np.sum(model_relative_likelihoods)


def evidence_ratios(prob_1, prob_2):
    return prob_1 / prob_2


def r2(y_true, y_model, x, logscaled=False):
    ss_res = RSS(y_true, y_model, x, logscaled=logscaled)
    ss_tot = RSS(y_true, np.mean(y_true), x, logscaled=logscaled)
    return 1 - ss_res / ss_tot


# decay types
def powerlaw_decay(p, x):
    return p["p_init"] * x ** (p["p_decay_const"]) + p["intercept"]


def exp_decay(p, x):
    return p["e_init"] * np.exp(-x * p["e_decay_const"]) + p["intercept"]


def pow_exp_decay(p, x):
    powerlaw = p["p_init"] * x ** (p["p_decay_const"])
    exponential = p["e_init"] * np.exp(-x * p["e_decay_const"])
    return powerlaw + exponential + p["intercept"]


# fitting model
def fit_model_iter(model, n_iter=10, **kwargs):
    """ re-fit model n_iter times and choose the best fit
    chooses method based upon best-fit
    """
    models = []
    AICs = []
    for iter in np.arange(n_iter):
        results_model = model.minimize(**kwargs)
        models.append(results_model)
        AICs.append(results_model.aic)
    return models[np.argmin(AICs)]


def model_res(p, x, y, fit, model):
    if fit == "lin":
        return residuals(y, model(p, x), x)
    else:
        return residuals(y, model(p, x), x, logscaled=True)


def curvature(y):
    y= y.astype('float64')
    return np.gradient(np.gradient(y)) / ((1+np.gradient(y)**2)**(3/2) )

def likelihood_ratio(llmin, llmax):
    return(2*(llmax-llmin))

def LRT(y, results1, model1, results2, model2, distances, logscaled = False):
    """ performs a likelihood ratio test for two lmfit models
    """
    # n samples
    n = len(distances)
    
    # n parameters for each
    k1 = len(results1.params)
    k2 = len(results2.params)
    
    # get model y
    y1 = get_y(model1, results1, distances)
    y2 = get_y(model2, results2, distances)
    
    # get likelihood
    LL1 = log_likelihood(n, y, y1, distances, logscaled=logscaled)
    LL2 = log_likelihood(n, y, y2, distances, logscaled=logscaled)
    
    # perform likelihood ratio test
    LR = likelihood_ratio(LL1,LL2)
    p = chi2.sf(LR, k2-k1)
    return p


def fit_models(
    distances,
    sig,
    fit="log",
    n_iter=1,
    method=["nelder", "leastsq", "least-squares"],
    p_power=None,
    p_exp=None,
    p_pow_exp=None,
    p_exp_exp=None,
):

    # add many takes input as a tuple (value, vary, min, max)
    if p_power is None:
        p_power = lmfit.Parameters()
        p_power.add_many(
            ("p_init", 0.5, True, 1e-10),
            ("p_decay_const", -0.5, True, -np.inf, -1e-10),
            ("intercept", 1e-5, True, 1e-10),
        )
    if p_exp is None:
        p_exp = lmfit.Parameters()
        p_exp.add_many(
            ("e_init", 0.5, True, 1e-10),
            ("e_decay_const", 0.1, True, 1e-10),
            ("intercept", 1e-5, True, 1e-10),
        )
    if p_pow_exp is None:
        p_pow_exp = lmfit.Parameters()
        p_pow_exp.add_many(
            ("e_init", 0.5, True, 1e-19),
            ("e_decay_const", 0.1, True, 1e-10),
            ("p_init", 0.5, True, 1e-10),
            ("p_decay_const", -0.5, True, -np.inf, -1e-10),
            ("intercept", 1e-5, True, 1e-10),
        )

    results_power_min = lmfit.Minimizer(
        model_res,
        p_power,
        fcn_args=(distances, sig, fit, powerlaw_decay),
        nan_policy="omit",
    )

    results_power = [
        fit_model_iter(results_power_min, n_iter=n_iter, **{"method": meth})
        for meth in method
    ]
    results_power = results_power[np.argmin([i.aic for i in results_power])]

    results_exp_min = lmfit.Minimizer(
        model_res, p_exp, fcn_args=(distances, sig, fit, exp_decay), nan_policy="omit"
    )
    results_exp = [
        fit_model_iter(results_exp_min, n_iter=n_iter, **{"method": meth})
        for meth in method
    ]
    results_exp = results_exp[np.argmin([i.aic for i in results_exp])]

    results_pow_exp_min = lmfit.Minimizer(
        model_res,
        p_pow_exp,
        fcn_args=(distances, sig, fit, pow_exp_decay),
        nan_policy="omit",
    )
    results_pow_exp = [
        fit_model_iter(results_pow_exp_min, n_iter=n_iter, **{"method": meth})
        for meth in method
    ]
    results_pow_exp = results_pow_exp[np.argmin([i.aic for i in results_pow_exp])]

    best_fit_model = np.array(["pow", "exp", "pow_exp", "concat"])[
        np.argmin([results_power.aic, results_exp.aic, results_pow_exp.aic])
    ]
    return results_power, results_exp, results_pow_exp, best_fit_model


def get_y(model, results, x):
    return model({i: results.params[i].value for i in results.params}, x)


def fit_results(
    sig,
    distances,
    results_exp,
    results_power,
    results_pow_exp,
    mask=True,
    logscaled=True,
):
    if mask:
        mask = sig > 0
        distances = distances[mask]
        sig = sig[mask]

    """ Get model fit results"""
    y_exp = get_y(exp_decay, results_exp, distances)
    y_concat = get_y(pow_exp_decay, results_pow_exp, distances)
    y_power = get_y(powerlaw_decay, results_power, distances)

    # calculate R2
    R2_exp = r2(sig, y_exp, distances, logscaled=logscaled)
    R2_concat = r2(sig, y_concat, distances, logscaled=logscaled)
    R2_power = r2(sig, y_power, distances, logscaled=logscaled)

    # calculate AIC
    AICc_exp = AICc(
        len(distances),
        len(results_exp.params),
        sig,
        y_exp,
        distances,
        logscaled=logscaled,
    )
    AICc_pow = AICc(
        len(distances),
        len(results_power.params),
        sig,
        y_power,
        distances,
        logscaled=logscaled,
    )
    AICc_concat = AICc(
        len(distances),
        len(results_pow_exp.params),
        sig,
        y_concat,
        distances,
        logscaled=logscaled,
    )
    return R2_exp, R2_concat, R2_power, AICc_exp, AICc_pow, AICc_concat
