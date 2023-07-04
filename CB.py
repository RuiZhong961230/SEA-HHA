"""
Hadi Bayzidi, Siamak Talatahari, Meysam Saraee, et al.
Social Network Search for Solving Engineering Optimization Problems[J].
Computational Intelligence and Neuroscience
"""

"""
Cantilever Beam
"""
def CB_obj(X):
    """
    :param X:
    0.01 <= x_i <= 100
    :return:
    """
    return 0.0624 * sum(X)


def CB_cons(X):
    con1 = 61 / (X[0] ** 3) + 37 / (X[1] ** 3) + 19 / (X[2] ** 3) + 7 / (X[3] ** 3) + 1 / (X[4] ** 3) - 1
    return [con1]

