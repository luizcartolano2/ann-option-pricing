# -*- coding: utf-8 -*-
"""
    NAME
        black_scholes
    DESCRIPTION
        Module that implement functions to calculate
        options price based on a generalized Black Scholes
        module.
    FUNCTIONS
        euro_vanilla_call
            Python implementation for a generic black scholes model to calculate
            call option values.
        euro_vanilla_put
            Python implementation for a generic black scholes model to calculate
            put option values.
    EXAMPLES
        Black Scholes: stock Options (no dividend yield)
            foreign_rate = risk_free_rate
        Merton Model: Stocks Index, stocks with a continuous dividend yields
            foreign_rate = risk_free_rate - dividend payment
        Commodities (default for both Call and Put)
            foreign_rate = 0
        FX Options (Garman Kohlhagen)
            foreign_rate = risk_free_rate - Foreign Risk Free Rate
"""
import numpy as np
from scipy.stats import norm


def check_inputs(spot, strike, tenor, risk_free_rate, sigma, foreign_rate):
    """
        Python implementation for  preliminary calculations for the BS Model.

        :param spot: price of the underlying asset on the valuation date.
        :param strike: strike, or exercise, price of the option.
        :param tenor: the time to expiration in years.
        :param risk_free_rate: expected return on a risk-free investment.
        :param sigma: volatility of the underlying security.
        :param foreign_rate: he risk free rate of the foreign currency.

        :return: the inputs as np.arrays
    """
    if hasattr(spot, 'shape') and len(spot.shape) != 0 and len(spot.shape) != 2:
        spot = spot.reshape((spot.shape[0], 1))

    if hasattr(strike, 'shape') and len(strike.shape) != 0 and len(strike.shape) != 2:
        strike = strike.reshape((strike.shape[0], 1))

    if hasattr(tenor, 'shape') and len(tenor.shape) != 0 and len(tenor.shape) != 2:
        tenor = tenor.reshape((tenor.shape[0], 1))

    if hasattr(risk_free_rate, 'shape') and len(risk_free_rate.shape) != 0 and len(risk_free_rate.shape) != 2:
        risk_free_rate = risk_free_rate.reshape((risk_free_rate.shape[0], 1))

    if hasattr(sigma, 'shape') and len(sigma.shape) != 0 and len(sigma.shape) != 2:
        sigma = sigma.reshape((sigma.shape[0], 1))

    if hasattr(foreign_rate, 'shape') and len(foreign_rate.shape) != 0 and len(foreign_rate.shape) != 2:
        foreign_rate = foreign_rate.reshape((foreign_rate.shape[0], 1))


    return spot, strike, tenor, risk_free_rate, sigma, foreign_rate


def create_preliminary_calculations(spot, strike, tenor, risk_free_rate, sigma, foreign_rate=0):
    """
        Python implementation for  preliminary calculations for the BS Model.

        :param spot: price of the underlying asset on the valuation date.
        :param strike: strike, or exercise, price of the option.
        :param tenor: the time to expiration in years.
        :param risk_free_rate: expected return on a risk-free investment.
        :param sigma: volatility of the underlying security.
        :param foreign_rate: he risk free rate of the foreign currency.

        :return: d1, d2
    """
    tenor_sqrt = np.sqrt(tenor)
    d1 = (np.log(spot / strike) + (foreign_rate + (sigma ** 2) / 2) * tenor) / (sigma * tenor_sqrt)
    d2 = d1 - sigma * tenor_sqrt

    return d1, d2


def euro_vanilla_call(spot, strike, tenor, risk_free_rate, sigma, foreign_rate=0):
    """
        Python implementation for a generic black scholes model to calculate
        call option values.

        :param spot: price of the underlying asset on the valuation date.
        :param strike: strike, or exercise, price of the option.
        :param tenor: the time to expiration in years.
        :param risk_free_rate: expected return on a risk-free investment.
        :param sigma: volatility of the underlying security.
        :param foreign_rate: he risk free rate of the foreign currency.

        :return: the price for the call option.
    """
    spot, strike, tenor, risk_free_rate, sigma, foreign_rate = check_inputs(spot, strike,
                                                                            tenor, risk_free_rate,
                                                                            sigma, foreign_rate)

    d1, d2 = create_preliminary_calculations(spot, strike, tenor,
                                             risk_free_rate, sigma,
                                             foreign_rate)

    call = spot * np.exp((foreign_rate - risk_free_rate) * tenor) * \
           norm.cdf(d1) - strike * np.exp(-risk_free_rate * tenor) * norm.cdf(d2)

    return call


def euro_vanilla_put(spot, strike, tenor, risk_free_rate, sigma, foreign_rate=0):
    """
        Python implementation for a generic black scholes model to calculate
        put option values.

        :param spot: price of the underlying asset on the valuation date.
        :param strike: strike, or exercise, price of the option.
        :param tenor: the time to expiration in years.
        :param risk_free_rate: expected return on a risk-free investment.
        :param sigma: volatility of the underlying security.
        :param foreign_rate: he risk free rate of the foreign currency.

        :return: the price for the put option.
    """
    spot, strike, tenor, risk_free_rate, sigma, foreign_rate = check_inputs(spot, strike,
                                                                            tenor, risk_free_rate,
                                                                            sigma, foreign_rate)

    d1, d2 = create_preliminary_calculations(spot, strike, tenor,
                                             risk_free_rate, sigma,
                                             foreign_rate)

    put = strike * np.exp(-risk_free_rate * tenor) * norm.cdf(-d2) - \
          (spot * np.exp((foreign_rate - risk_free_rate) * tenor) * norm.cdf(-d1))

    return put
