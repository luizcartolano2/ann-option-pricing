import warnings

import numpy as np
from scipy.stats import norm

warnings.filterwarnings("ignore")


class ImpliedVolatility:
    """
    """
    def __init__(self, risk_free, tol, max_iter, foreign_rate=0):
        self.__risk_free = risk_free
        self.__tol = tol
        self.__max_iter = max_iter
        self.__foreign_rate = foreign_rate

    @staticmethod
    def __check_input(param):
        """
            :param param:
            :return:
        """
        if hasattr(param, 'shape') and len(param.shape) != 0 and len(param.shape) != 2:
            param = param.reshape((param.shape[0], 1))

        return param

    def __black_d1(self, spot, strike, tenor, sigma):
        """
            Python for d1 preliminary calculation for black scholes model.

            :param spot: price of the underlying asset on the valuation date.
            :param strike: strike, or exercise, price of the option.
            :param tenor: the time to expiration in years.
            :param sigma: volatility of the underlying security.

            :return: the price for the call option.
        """
        tenor_sqrt = np.sqrt(tenor)
        foreign_rate = self.__foreign_rate

        black_d1_calc = (np.log(spot / strike) + (foreign_rate + (sigma ** 2) / 2) * tenor) / (sigma * tenor_sqrt)

        return black_d1_calc

    def __black_d2(self, spot, strike, tenor, sigma):
        """
            Python for d2 preliminary calculation for black scholes model.

            :param spot: price of the underlying asset on the valuation date.
            :param strike: strike, or exercise, price of the option.
            :param tenor: the time to expiration in years.
            :param sigma: volatility of the underlying security.

            :return: the price for the call option.
        """
        tenor_sqrt = np.sqrt(tenor)

        black_d2_calc = self.__black_d1(spot, strike, tenor, sigma) - sigma * tenor_sqrt

        return black_d2_calc

    def euro_vanilla_call(self, spot, strike, tenor, sigma):
        """
            Python implementation for a generic black scholes model to calculate
            call option values.

            :param spot: price of the underlying asset on the valuation date.
            :param strike: strike, or exercise, price of the option.
            :param tenor: the time to expiration in years.
            :param sigma: volatility of the underlying security.

            :return: the price for the call option.
        """
        risk_free_rate = self.__risk_free
        foreign_rate = self.__foreign_rate

        spot = self.__check_input(spot)
        strike = self.__check_input(strike)
        tenor = self.__check_input(tenor)
        risk_free_rate = self.__check_input(risk_free_rate)
        sigma = self.__check_input(sigma)
        foreign_rate = self.__check_input(foreign_rate)

        black_d1_calc = self.__black_d1(spot, strike, tenor, sigma)
        black_d2_calc = self.__black_d2(spot, strike, tenor, sigma)

        call = spot * np.exp((foreign_rate - risk_free_rate) * tenor) * \
               norm.cdf(black_d1_calc) - strike * np.exp(-risk_free_rate * tenor) * \
               norm.cdf(black_d2_calc)

        return call

    def euro_vanilla_put(self, spot, strike, tenor, sigma):
        """
            Python implementation for a generic black scholes model to calculate
            put option values.

            :param spot: price of the underlying asset on the valuation date.
            :param strike: strike, or exercise, price of the option.
            :param tenor: the time to expiration in years.
            :param sigma: volatility of the underlying security.

            :return: the price for the put option.
        """
        risk_free_rate = self.__risk_free
        foreign_rate = self.__foreign_rate

        spot = self.__check_input(spot)
        strike = self.__check_input(strike)
        tenor = self.__check_input(tenor)
        risk_free_rate = self.__check_input(risk_free_rate)
        sigma = self.__check_input(sigma)
        foreign_rate = self.__check_input(foreign_rate)

        black_d1_calc = self.__black_d1(spot, strike, tenor, sigma)
        black_d2_calc = self.__black_d2(spot, strike, tenor, sigma)

        put = strike * np.exp(-risk_free_rate * tenor) * norm.cdf(-black_d2_calc) - \
              (spot * np.exp((foreign_rate - risk_free_rate) * tenor) * norm.cdf(-black_d1_calc))

        return put

    def calc_opt_price(self, underlying, call_put, strike, tenor, sigma):
        """
            :param underlying:
            :param call_put:
            :param strike:
            :param tenor:
            :param sigma:

            :return:
        """
        risk_free = self.__risk_free
        call_put = self.__check_input(call_put)

        # calcula os precos de put e call pelo modelo de black-scholes
        call_prices = self.euro_vanilla_call(underlying, strike, tenor, sigma)
        put_prices = self.euro_vanilla_put(underlying, strike, tenor, sigma)

        # trata os precos calculados pelo modelo de black-scholes separadamente
        call_prices = call_put * call_prices
        put_prices = np.logical_not(call_put) * 1 * put_prices

        # junta os prices de call e put e um unico vetor
        options_price = call_prices + put_prices

        return options_price

    def implied_volatility_bissec(self, prices, underlying, strike, call_put, tenor):
        """
            :param prices: price vector
            :param underlying: underlying vector
            :param strike: strike vector
            :param call_put: call/put vector
            :param tenor: tentative vol vector

            :return i: number of iterations
            :return temp: tolerance
            :return vol: volatility
        """
        func = lambda vol_sig: self.calc_opt_price(underlying, call_put, strike, tenor, vol_sig)

        # se o valor de vega for muito baixo
        # encontramos a vol pelo metodo da bisec
        sigma_a = .1 * np.ones(strike.shape)
        sigma_b = 2. * np.ones(strike.shape)

        middle_pt = sigma_a

        i = 1
        while i < self.__max_iter and (abs(sigma_b - sigma_a) > self.__tol).all():
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
