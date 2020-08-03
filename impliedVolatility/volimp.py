import numpy as np
from .black_scholes_model import *
import warnings

warnings.filterwarnings("ignore")


def calc_opt_price(underlying, call_put, strike, tenor, r, sigma):
    # calcula os precos de put e call pelo modelo de black-scholes
    f_call = euro_vanilla_call(underlying, strike, tenor, r, sigma)
    f_put = euro_vanilla_put(underlying, strike, tenor, r, sigma)

    # trata os precos calculados pelo modelo de black-scholes separadamente
    f_call = call_put * f_call
    f_put = np.logical_not(call_put) * 1 * f_put
    # junta os prices de call e put e um unico vetor
    f = f_call + f_put

    return f


def volimp(prices, underlying, strike, call_put, r, tenor, tol, max_iter):
    """
        :input: prices - price vector
        :input: underlying - underlying vector
        :input: strike - strike vector
        :input: call_put - call/put vector
        :input: r - interest rate vector
        :input: tenor - tentative vol vector
        :input: tol - solution tolerance
        :input: max_iter - maximum iterations allowed

        :return: i - number of iterations
        :return: temp - tolerance
        :return: vol - volatility
    """

    func = lambda vol_sig: calc_opt_price(underlying, call_put, strike, tenor, r, vol_sig)

    # tentative vol
    sigma = .25 * np.ones(strike.shape)

    # se o valor de vega for muito baixo
    # encontramos a vol pelo metodo da bisec
    sigma_a = .1 * np.ones(strike.shape)
    sigma_b = 2. * np.ones(strike.shape)

    middle_pt = sigma_a

    i = 1
    while i < max_iter and (abs(sigma_b - sigma_a) > tol).all():
        # find middle point
        middle_pt = (sigma_a + sigma_b) / 2

        # first we check all the strikes where the
        # condition f(middle_point) * f(sigma_a) < 0
        # satisfies
        temp_sgb = ((func(middle_pt) - prices) * (func(sigma_a) - prices)) < 0
        # check which of the middle_point is root
        temp_root = ((func(middle_pt) - prices) * (func(sigma_a) - prices)) == 0

        # in the update of the bisect method, if
        # f(middle_point) * f(sigma_a) < 0 we update
        # the intervals such that sigma_a = sigma_a and
        # sigma_b = middle_point, otherwise, we make,
        # sigma_a = middle_point and sigma_b = sigma_b

        # since we are working with a vector of values
        # to update sigma_a we multiply all the True values
        # in the conditions vector by middle_pt, and all the
        # False values by the sigma_a, we also multiply the True
        # values from the root condition by sigma_a to keep that
        # values
        sigma_a = np.logical_not(temp_sgb) * middle_pt + temp_sgb * sigma_a + temp_root * sigma_a

        # to update the sigma_b we use the same logic
        # from sigma_a, just reversing the True/False
        # multiplies for the condition vector
        sigma_b = temp_sgb * middle_pt + np.logical_not(temp_sgb) * sigma_b + temp_root * sigma_b

        # update the counter
        i += 1

    temp = 1
    vol = middle_pt

    return i, temp, vol
