"""
Seyedali Mirjalili, Seyed Mohammad Mirjalili, Andrew Lewis,
Grey Wolf Optimizer,
Advances in Engineering Software,
Volume 69,
2014,
Pages 46-61,
https://doi.org/10.1016/j.advengsoft.2013.12.007.
"""


import numpy as np

"""
Pressure Vessel Design
"""
def PVD_obj(X):
    """
    :param X:
    0 <= X[0] <= 99,
    0 <= X[1] <= 99,
    10 <= X[2] <= 200,
    10 <= X[3] <= 200
    :return:
    """
    return 0.6224 * X[0] * X[2] * X[3] + 1.7781 * X[1] * X[2] ** 2 + 3.1661 * X[0] ** 2 * X[3] + 19.84 * X[0] ** 2 * X[2]


def PVD_cons(X):
    con1 = -X[0] + 0.0193 * X[2]
    con2 = -X[1] + 0.00954 * X[2]
    con3 = -np.pi * X[2] ** 2 * X[3] - 4 / 3 * np.pi * X[2] ** 3 + 1296000
    con4 = X[3] - 240
    return [con1, con2, con3, con4]

