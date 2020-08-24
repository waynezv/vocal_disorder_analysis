# -*- coding: utf-8 -*-
import numpy as np


def optim_grad_step(alpha, beta, delta, d_alpha, d_beta, d_delta, stepsize=0.01):
    """
    Perform one step of gradient descent for model parameters.
    """
    alpha = alpha - stepsize * d_alpha
    beta = beta - stepsize * d_beta
    delta = delta - stepsize * d_delta
    return alpha, beta, delta


def optim_adapt_step(alpha, beta, delta, d_alpha, d_beta, d_delta, default_step=0.01):
    """
    Perform one step of gradient descent for model parameters.
    Stepsize is adaptive.
    """
    stepsize = default_step / np.max([d_alpha, d_beta, d_delta])

    if (alpha - stepsize * d_alpha) > 0 and (alpha - stepsize * d_alpha) < 2:
        alpha = alpha - stepsize * d_alpha

    if (beta - stepsize * d_beta) > 0 and (beta - stepsize * d_beta) < 2:
        beta = beta - stepsize * d_beta

    if (delta - stepsize * d_delta) > 0 and (delta - stepsize * d_delta) < 2:
        delta = delta - stepsize * d_delta

    return alpha, beta, delta


def optim_adam(p, dp, m_t, v_t, itr, eta=0.01, beta_1=0.9, beta_2=0.999, eps=1e-8):
    """
    Perform Adam update.

    Parameters
    ----------
    p: float
        Parameter.
    dp: float
        Gradient.
    m_t: float
        Moving average of gradient.
    v_t: float
        Moving average of gradient squared.
    itr: float
        Iteration.
    eta: float
        Learning rate.
    beta_1: float
        Decay for gradient.
    beta_2: float
        Decay for gradient squared.
    eps: float
        Tolerance.

    Returns
    -------
    p: float
        Updated parameter.
    """
    m_t = beta_1 * m_t + (1 - beta_1) * dp
    v_t = beta_2 * v_t + (1 - beta_2) * (dp * dp)
    m_cap = m_t / (1 - (beta_1 ** itr))  # correct bias
    v_cap = v_t / (1 - (beta_2 ** itr))
    p = p - (eta * m_cap) / (np.sqrt(v_cap) + eps)
    return p
