"""
Seyedali Mirjalili, Seyed Mohammad Mirjalili, Andrew Lewis,
Grey Wolf Optimizer,
Advances in Engineering Software,
Volume 69,
2014,
Pages 46-61,
https://doi.org/10.1016/j.advengsoft.2013.12.007.
"""


"""
Compression Spring Design
"""
def CSD_obj(X):
    """
    :param X:
    0.05 <= X[0] <= 2,
    0.25 <= X[1] <= 1.3,
    2 <= X[2] <= 15
    :return:
    """
    return (X[2] + 2) * X[1] * X[0] ** 2


def CSD_cons(X):
    """
    :return: All cons should be minus than 0
    """
    con1 = 1 - (X[1] ** 3 * X[2]) / (71785 * X[0] ** 4)
    con2 = (4 * X[1] ** 2 - X[0] * X[1]) / (12566 * (X[1] * X[0] ** 3 - X[0] ** 4)) + 1 / (5108 * X[0] ** 2) - 1
    con3 = 1 - 140.45 * X[0] / (X[1] ** 2 * X[2])
    con4 = (X[0] + X[1]) / 1.5 - 1
    return [con1, con2, con3, con4]
