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


warnings.filterwarnings('ignore')


POPULATION_SIZE = 100
DIMENSION_NUM = 10
LOWER_BOUNDARY = -100
UPPER_BOUNDARY = 100
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


def Evaluation(indi, bench):
    global Fun_num
    return bench.Y(indi, Fun_num)


def CheckIndi(Indi):
    range_width = UPPER_BOUNDARY - LOWER_BOUNDARY
    for i in range(DIMENSION_NUM):
        if Indi[i] > UPPER_BOUNDARY:
            n = int((Indi[i] - UPPER_BOUNDARY) / range_width)
            mirrorRange = (Indi[i] - UPPER_BOUNDARY) - (n * range_width)
            Indi[i] = UPPER_BOUNDARY - mirrorRange
        elif Indi[i] < LOWER_BOUNDARY:
            n = int((LOWER_BOUNDARY - Indi[i]) / range_width)
            mirrorRange = (LOWER_BOUNDARY - Indi[i]) - (n * range_width)
            Indi[i] = LOWER_BOUNDARY + mirrorRange
        else:
            pass


def Space(data):
    data = np.array(data)
    limit_scale = []
    for i in range(len(data[0])):
        d = data[:, i]
        limit_scale.append([min(d), max(d)])
    return np.array(limit_scale)


def Initialization(bench):
    global Archive_solution, Archive_fitness, Population, Population_fitness, INITIAL_SIZE
    Archive_solution[0:INITIAL_SIZE] = LOWER_BOUNDARY + (UPPER_BOUNDARY - LOWER_BOUNDARY) * lhs(DIMENSION_NUM, samples=INITIAL_SIZE)
    for i in range(INITIAL_SIZE):
        Archive_fitness[i] = Evaluation(Archive_solution[i], bench)
    order = np.argsort(Archive_fitness[0:INITIAL_SIZE])
    Population = Archive_solution[order[0:POPULATION_SIZE]]
    Population_fitness = Archive_fitness[order[0:POPULATION_SIZE]]


def LSearch(i, g, bench):
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
    Offspring_fitness[i] = Evaluation(Offspring[i], bench)
    Archive_solution[g] = Offspring[i]
    Archive_fitness[g] = Offspring_fitness[i]


def DSearch(i, g, bench):
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
    Offspring_fitness[i] = Evaluation(Offspring[i], bench)
    Archive_solution[g] = Offspring[i]
    Archive_fitness[g] = Offspring_fitness[i]


def Reinitialization(i, g, bench):
    global Population, Population_fitness, Archive_solution, Archive_fitness, Offspring, Offspring_fitness
    for t in range(DIMENSION_NUM):
        Offspring[i][t] = np.random.uniform(LOWER_BOUNDARY, UPPER_BOUNDARY)
    Offspring_fitness[i] = Evaluation(Offspring[i], bench)
    Archive_solution[g] = Offspring[i]
    Archive_fitness[g] = Offspring_fitness[i]


def Loss(model, X_test, y_test):
    y_predict = model.predict(X_test)
    return mean_absolute_error(y_predict, y_test)


def SurrogateEstimation(i, g, bench):
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
    Offspring_fitness[i] = Evaluation(Offspring[i], bench)
    Archive_solution[g] = Offspring[i]
    Archive_fitness[g] = Offspring_fitness[i]


def SEAHHA(g, bench):
    Archives = [LSearch, DSearch, SurrogateEstimation, Reinitialization]
    for i in range(POPULATION_SIZE):
        Strategy = np.random.choice(Archives, p=[0.33, 0.33, 0.33, 0.01])
        Strategy(i, g, bench)
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


def RunSEAHHA():
    global Fun_num, Population_fitness, MAX_FITNESS_EVALUATION_NUM
    All_Trial_Best = []
    All_Trial_Best_Evaluation_Num = []
    bench = CEC2013(DIMENSION_NUM)
    for t in range(REPETITION_NUM):
        np.random.seed(2022 + 88 * t)
        Best_list = []
        Best_Evaluation_Num_list = []
        Initialization(bench)
        Best_list.append(min(Population_fitness))
        g = INITIAL_SIZE
        while g < MAX_FITNESS_EVALUATION_NUM:
            g = SEAHHA(g, bench)
            Best_list.append(min(Population_fitness[0:g]))
        print("min: ", Best_list[-1])
        All_Trial_Best.append(Best_list)
        All_Trial_Best_Evaluation_Num.append(Best_Evaluation_Num_list)
    np.savetxt('./SEA-HHA_Data/F{}_{}D.csv'.format(Fun_num, DIMENSION_NUM), All_Trial_Best, delimiter=",")


def main(Dim):
    global Fun_num, DIMENSION_NUM, Population, Population_fitness, MAX_FITNESS_EVALUATION_NUM, Archive_solution, Offspring

    DIMENSION_NUM = Dim
    Population = np.zeros((POPULATION_SIZE, DIMENSION_NUM))
    Archive_solution = np.zeros((MAX_FITNESS_EVALUATION_NUM, DIMENSION_NUM))
    Offspring = np.zeros((POPULATION_SIZE, DIMENSION_NUM))

    for i in range(1, 29):
        Fun_num = i
        RunSEAHHA()


if __name__ == "__main__":
    if os.path.exists('./SEA-HHA_Data') == False:
        os.makedirs('./SEA-HHA_Data')
    Dim = 10
    main(Dim)