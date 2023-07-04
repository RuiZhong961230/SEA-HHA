import os
import numpy as np
from CEC2013 import CEC2013
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as CK
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from DE import DE, RandomSearch
import warnings
from pyDOE2 import lhs
from sklearn.metrics import mean_absolute_error
from CB import CB_obj, CB_cons
from CSD import CSD_obj, CSD_cons
from PVD import PVD_obj, PVD_cons


warnings.filterwarnings('ignore')


MAX_FITNESS = 9999999999999999
POPULATION_SIZE = 100
DIMENSION_NUM = 10
LOWER_BOUNDARY = []
UPPER_BOUNDARY = []
REPETITION_NUM = 30
MAX_FITNESS_EVALUATION_NUM = 1000
INITIAL_SIZE = 100

Archive_solution = np.zeros((MAX_FITNESS_EVALUATION_NUM, DIMENSION_NUM))
Archive_fitness = np.zeros(MAX_FITNESS_EVALUATION_NUM)

Population = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
Population_fitness = np.zeros(POPULATION_SIZE)

Offspring = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
Offspring_fitness = np.zeros(POPULATION_SIZE)

K = 100
Fun_num = 1
F = 0.8
R = 2


def Evaluate(func, cons, X):
    penalty = cons(X)
    if isNegative(penalty) == False:
        obj = MAX_FITNESS
    else:
        obj = func(X)
    return obj


def isNegative(penalty):
    for i in range(len(penalty)):
        if penalty[i] > 0:
            return False
    return True


def CheckIndi(Indi):
    global UPPER_BOUNDARY, LOWER_BOUNDARY
    for i in range(DIMENSION_NUM):
        range_width = UPPER_BOUNDARY[i] - LOWER_BOUNDARY[i]
        if Indi[i] > UPPER_BOUNDARY[i]:
            n = int((Indi[i] - UPPER_BOUNDARY[i]) / range_width)
            mirrorRange = (Indi[i] - UPPER_BOUNDARY[i]) - (n * range_width)
            Indi[i] = UPPER_BOUNDARY[i] - mirrorRange
        elif Indi[i] < LOWER_BOUNDARY[i]:
            n = int((LOWER_BOUNDARY[i] - Indi[i]) / range_width)
            mirrorRange = (LOWER_BOUNDARY[i] - Indi[i]) - (n * range_width)
            Indi[i] = LOWER_BOUNDARY[i] + mirrorRange
        else:
            pass


def Space(data):
    data = np.array(data)
    limit_scale = []
    for i in range(len(data[0])):
        d = data[:, i]
        limit_scale.append([min(d), max(d)])
    return np.array(limit_scale)


def Initialization(func, cons):
    global Archive_solution, Archive_fitness, Population, Population_fitness, INITIAL_SIZE
    Archive_solution[0:INITIAL_SIZE] = lhs(DIMENSION_NUM, samples=INITIAL_SIZE)
    for i in range(INITIAL_SIZE):
        for j in range(DIMENSION_NUM):
            Archive_solution[i][j] *= LOWER_BOUNDARY[j] + np.random.rand() * (UPPER_BOUNDARY[j] - LOWER_BOUNDARY[j])
        Archive_fitness[i] = Evaluate(func, cons, Archive_solution[i])
    order = np.argsort(Archive_fitness[0:INITIAL_SIZE])
    Population = Archive_solution[order[0:POPULATION_SIZE]]
    Population_fitness = Archive_fitness[order[0:POPULATION_SIZE]]


def LSearch(i, g, func, cons):
    global Population, Population_fitness, Archive_solution, Archive_fitness, Offspring, Offspring_fitness
    r = np.random.rand()
    if r < 1 / 3:
        X_base = Population[int(np.random.randint(0, POPULATION_SIZE))]
    elif r < 2 / 3:
        X_base = Population[i]
    else:
        X_base = Population[np.argmin(Population_fitness)]

    # X_base = Population[np.argmin(Population_fitness)]
    for j in range(DIMENSION_NUM):
        X_base[j] += R * np.random.uniform(-1, 1)
    CheckIndi(X_base)
    Offspring[i] = X_base
    Offspring_fitness[i] = Evaluate(func, cons, Offspring[i])
    Archive_solution[g] = Offspring[i]
    Archive_fitness[g] = Offspring_fitness[i]


def DSearch(i, g, func, cons):
    global Population, Population_fitness, Archive_solution, Archive_fitness, Offspring, Offspring_fitness
    r1, r2, r3 = np.random.choice(list(range(0, POPULATION_SIZE)), 3, replace=False)
    r = np.random.rand()
    if r < 1 / 3:
        X_base = Population[r1]
    elif r < 2 / 3:
        X_base = Population[np.argmin(Population_fitness)]
    else:
        X_base = Population[np.argmin(Population_fitness)]
    # X_base = Population[np.argmin(Population_fitness)]
    Vector = Population[r2] - Population[r3]
    for j in range(DIMENSION_NUM):
        Vector[j] *= F * np.random.uniform(-1, 1)
    Offspring[i] = X_base + Vector
    CheckIndi(Offspring[i])
    Offspring_fitness[i] = Evaluate(func, cons, Offspring[i])
    Archive_solution[g] = Offspring[i]
    Archive_fitness[g] = Offspring_fitness[i]


def Reinitialization(i, g, func, cons):
    global Population, Population_fitness, Archive_solution, Archive_fitness, Offspring, Offspring_fitness
    for t in range(DIMENSION_NUM):
        Offspring[i][t] = np.random.uniform(LOWER_BOUNDARY[t], UPPER_BOUNDARY[t])
    Offspring_fitness[i] = Evaluate(func, cons, Offspring[i])
    Archive_solution[g] = Offspring[i]
    Archive_fitness[g] = Offspring_fitness[i]


def Loss(model, X_test, y_test):
    y_predict = model.predict(X_test)
    return mean_absolute_error(y_predict, y_test)


def SurrogateEstimation(i, g, func, cons):
    global Archive_solution, Archive_fitness, Population, Population_fitness, Offspring, Offspring_fitness
    r = np.random.rand()
    if r < 1 / 3:  # Global model
        subPop = Archive_solution[0:g]
        subFit = Archive_fitness[0:g].reshape(-1, 1)
    elif r < 2 / 3:  # Recent model
        subPop = Archive_solution[g - K:g]
        subFit = Archive_fitness[g - K:g]
    else:  # Neighbor model
        X_best = Population[np.argmin(Population_fitness)]
        dis = np.zeros(g)
        for j in range(g):
            dis[j] = sum(abs(Archive_solution[j] - X_best))
        order = np.argsort(dis)[0: K]
        subPop = Archive_solution[order]
        subFit = Archive_fitness[order]

    space = Space(subPop)
    X_train, X_test, y_train, y_test = train_test_split(subPop, subFit, test_size=0.2)
    poly = PolynomialFeatures(degree=3)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)

    linear = LinearRegression()
    linear.fit(X_train_poly, y_train)

    gpr = GaussianProcessRegressor(alpha=5, n_restarts_optimizer=20, kernel=CK(1.0, (1e-4, 1e4)) * RBF(10, (1e-4, 1e4)))
    gpr.fit(X_train, y_train)

    svr = SVR()
    svr.fit(X_train, y_train)

    Models = [linear, gpr, svr]
    LOSS = [Loss(linear, X_test_poly, y_test), Loss(gpr, X_test, y_test), Loss(svr, X_test, y_test)]

    best_model = Models[np.argmin(LOSS)]
    flag = np.argmin(LOSS)
    if flag == 0:
        Offspring[i] = RandomSearch(best_model, poly, space)
    else:
        de = DE(best_model.predict, 50, DIMENSION_NUM, space[:, 0], space[:, 1], initX=subPop)
        Offspring[i] = de.run()

    CheckIndi(Offspring[i])
    Offspring_fitness[i] = Evaluate(func, cons, Offspring[i])
    Archive_solution[g] = Offspring[i]
    Archive_fitness[g] = Offspring_fitness[i]


def SEAHHA(g, func, cons):
    Archives = [LSearch, DSearch, SurrogateEstimation, Reinitialization]
    for i in range(POPULATION_SIZE):
        Strategy = np.random.choice(Archives, p=[0.33, 0.33, 0.33, 0.01])
        Strategy(i, g, func, cons)
        g += 1
    bestSelection()
    return g


def bestSelection():
    global Population, Population_fitness, Offspring, Offspring_fitness
    temp = np.vstack((Population, Offspring))
    temp_fitness = np.hstack((Population_fitness, Offspring_fitness))
    tmp = list(map(list, zip(range(len(temp_fitness)), temp_fitness)))
    small = sorted(tmp, key=lambda x: x[1], reverse=False)
    for i in range(POPULATION_SIZE):
        key, _ = small[i]
        Population_fitness[i] = temp_fitness[key]
        Population[i] = temp[key].copy()


def RunSEAHHA(prob, func, cons):
    global Fun_num, Population_fitness, MAX_FITNESS_EVALUATION_NUM
    All_Trial_Best = []
    Best_solution = None
    Best_fitness = float("inf")
    for t in range(REPETITION_NUM):
        np.random.seed(2022 + 88 * t)
        Best_list = []
        Initialization(func, cons)
        Best_list.append(min(Population_fitness))
        g = INITIAL_SIZE
        while g < MAX_FITNESS_EVALUATION_NUM:
            g = SEAHHA(g, func, cons)
            min_fit = min(Population_fitness)
            Best_list.append(min_fit)
            if min_fit < Best_fitness:
                Best_fitness = min_fit
                Best_solution = Population[np.argmin(Population_fitness)]
        print("min: ", Best_list[-1])
        All_Trial_Best.append(Best_list)
    np.savetxt('./SEA_HHA_Engineer_Data/' + prob + '.csv', All_Trial_Best, delimiter=",")
    np.savetxt('./SEA_HHA_Engineer_solution/' + prob + '.csv', [Best_solution], delimiter=",")


def main(prob, dim, func, cons, bound):
    global Fun_num, DIMENSION_NUM, Population, Population_fitness, MAX_FITNESS_EVALUATION_NUM, Archive_solution, Offspring
    global LOWER_BOUNDARY, UPPER_BOUNDARY

    DIMENSION_NUM = dim
    LOWER_BOUNDARY = bound[0]
    UPPER_BOUNDARY = bound[1]
    Population = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
    Archive_solution = np.zeros((MAX_FITNESS_EVALUATION_NUM, DIMENSION_NUM))
    Offspring = np.zeros((POPULATION_SIZE, DIMENSION_NUM))

    RunSEAHHA(prob, func, cons)


if __name__ == "__main__":
    if os.path.exists('./SEA_HHA_Engineer_Data') == False:
        os.makedirs('./SEA_HHA_Engineer_Data')
    Problems = ["CB", "PVD", "CSD"]
    Dims = [5, 4, 3]
    Funcs = [CB_obj, PVD_obj, CSD_obj]
    Cons = [CB_cons, PVD_cons, CSD_cons]
    Bounds = [
        [[0.01, 0.01, 0.01, 0.01, 0.01], [100, 100, 100, 100, 100]],
        [[0, 0, 10, 10], [99, 99, 200, 200]],
        [[0.05, 0.25, 2], [2, 1.3, 15]]
    ]
    for i in range(1, len(Problems)):
        # for i in range(1):
        main(Problems[i], Dims[i], Funcs[i], Cons[i], Bounds[i])