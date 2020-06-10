# Код для prochack турнира

import os
os.getcwd() # "/Users/nikitabaramiya/"
os.chdir("/Users/nikitabaramiya/Desktop/prohack_dataset")

from collections import Counter
# для анализа времени работы алгоритмов
import time
from datetime import datetime
# для обработки данных
import numpy as np
import pandas as pd
# для визуализации данных
import seaborn as sns
import matplotlib.pyplot as plt

# для заполнения пропусков и стандартизации
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor # также используем в ансамбле
from sklearn.neighbors import KNeighborsRegressor

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# предсказательные модели
from sklearn.linear_model import ElasticNetCV
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV
# ансамбли
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import StackingRegressor
# ускоренная версия градиентного бустинга
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingRegressor

# для построения и оценивания моделей
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# XGBoost (мб доберёмся)
# import xgboost as xgb


# импортируем данные
df_train = pd.read_csv('train.csv')
df_train.shape # (3865, 80)
df_test = pd.read_csv('test.csv')
df_test.shape # (890, 79)
df_sample_submit = pd.read_csv('sample_submit.csv')
df_sample_submit.shape # (890, 3)


# смотрим уникальные звёзды в тренировке и тесте
galaxies_train = set(df_train.galaxy)
galaxies_test = set(df_test.galaxy)

galaxies_test - galaxies_train # set()
galaxies_train - galaxies_test # {'Andromeda XXIV', 'Andromeda XVIII[60]', 'Triangulum Galaxy (M33)',
# 'Andromeda XXII[57]', 'Tucana Dwarf', 'Andromeda XII', 'Andromeda XIX[60]', 'NGC 5253', 'Hercules Dwarf'}

galaxies_train_dict = Counter(df_train.galaxy)
galaxies_train_dict.most_common()

# в тренировке больше звёзд, это хорошо, попробуем использовать дамми звёзд
dummy_train = pd.get_dummies(df_train['galaxy'])
df_train = pd.concat([df_train, dummy_train], axis = 1)
df_train.drop(columns='galaxy', inplace=True)

dummy_test = pd.get_dummies(df_test['galaxy'])
df_test = pd.concat([df_test, dummy_test], axis = 1)
df_test.drop(columns='galaxy', inplace=True)

# уберём целевую переменную
y_train = df_train['y']
X_train = df_train.drop(columns = ['y'])
X_train.shape # (3865, 259)
df_test.shape # (890, 250)

# дозаполним отсутсвующими столбцами
X_test = df_test.copy()
for col in list(galaxies_train - galaxies_test):
    X_test[col] = 0

X_test.shape # (890, 259)

# посмотрим, какая у нас целевая переменная
y_train.describe()
sns.distplot(y_train)
plt.show()

# смотрим, чего за дела с пустыми значениями
X_train.isna().sum().sort_values(ascending = False).head(20)


# попробуем монстр алгоритм
# лист заполнителей
imputers = [
    BayesianRidge(),
    DecisionTreeRegressor(max_features = 'auto', random_state = 0),
    KNeighborsRegressor(n_neighbors = 15),
    KNeighborsRegressor(n_neighbors = 7)
]

# возможный лист стандартизовшиков
# transformers = [
#     StandardScaler(),
#     MinMaxScaler(),
#     None
# ]

# лист оценщиков
estimators = [
    ElasticNetCV(),
    AdaBoostRegressor(),
    RandomForestRegressor(),
    HistGradientBoostingRegressor()
]

# лист гиперпараметров
tuned_parameters = [
    {'normalize': [True, False], 'l1_ratio': [.05, .1, .2, .5, .7, .9, .95, .99, 1], 'max_iter': [1500]},
    {'n_estimators': [200], 'random_state': [0]},
    {'n_estimators': [200], 'random_state': [0], 'max_depth': [*range(2, 6)]},
    {'max_iter': [200], 'random_state': [0], 'max_depth': [*range(2, 10)]}
]

# для кросс-валидации
k_fold = KFold(n_splits = 5, shuffle = True, random_state = 0)

# пытаемся найти лучший заполнитель и итоговый алгоритм
i = 1
scores = pd.DataFrame()
for impute_estimator in imputers:
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    time = datetime.now()
    print("Imputer: {}".format(impute_estimator.__class__.__name__))
    imputer = IterativeImputer(random_state = 0, estimator = impute_estimator)
    X_train_fill = imputer.fit_transform(X_train)
    print("Time: {}".format(datetime.now() - time))
    print("----------------------------------------------------------")
    for t, br_estimator in enumerate(estimators):
        time = datetime.now()
        print("Estimator: {}".format(br_estimator.__class__.__name__))
        # estimator = make_pipeline(, br_estimator)
        r2 = cross_val_score(br_estimator, X_train_fill, y_train, \
                        scoring = 'r2', cv = k_fold) # 'neg_mean_squared_error'
        scores["{0}: {1} + {2}".format(i, impute_estimator.__class__.__name__, \
                                        br_estimator.__class__.__name__)] = r2
        i += 1
        print("Score: {0:.2f}".format(np.mean(r2)))
        print("Time: {}".format(datetime.now() - time))
        print("----------------------------------------------------------")


    # for trans in transformers:
    #     if trans == None:
    #         print(trans)
    #     else:
    #         print(trans)
    #         X_train_fill = trans.fit_transform(X_train_fill)

scores.mean().sort_values()

# 5: DecisionTreeRegressor + ElasticNetCV                     0.387624
# 9: KNeighborsRegressor + ElasticNetCV                       0.431615
# 13: KNeighborsRegressor + ElasticNetCV                      0.437450
# 1: BayesianRidge + ElasticNetCV                             0.451565
# 6: DecisionTreeRegressor + AdaBoostRegressor                0.542737
# 10: KNeighborsRegressor + AdaBoostRegressor                 0.596777
# 14: KNeighborsRegressor + AdaBoostRegressor                 0.607979
# 2: BayesianRidge + AdaBoostRegressor                        0.663289
# 16: KNeighborsRegressor + HistGradientBoostingRegressor     0.832864
# 12: KNeighborsRegressor + HistGradientBoostingRegressor     0.833407
# 8: DecisionTreeRegressor + HistGradientBoostingRegressor    0.842782
# 7: DecisionTreeRegressor + RandomForestRegressor            0.857400
# 4: BayesianRidge + HistGradientBoostingRegressor            0.866520
# 15: KNeighborsRegressor + RandomForestRegressor             0.878828
# 11: KNeighborsRegressor + RandomForestRegressor             0.884269
# 3: BayesianRidge + RandomForestRegressor                    0.908524


# пытаемся улучшить предсказание поиском лучших параметров (пока среди тех те комбинаций и алгоритмов)
i = 0
all_info = [np.nan for i in range(len(imputers)*len(estimators))]
grid_scores = [np.inf for i in range(len(imputers)*len(estimators))]
best_params = [np.nan for i in range(len(imputers)*len(estimators))]
for impute_estimator in imputers:
    print("----------------------------------------------------------")
    print("----------------------------------------------------------")
    time = datetime.now()
    print("Imputer: {}".format(impute_estimator.__class__.__name__))
    imputer = IterativeImputer(random_state = 0, estimator = impute_estimator)
    X_train_fill = imputer.fit_transform(X_train)
    print("Time: {}".format(datetime.now() - time))
    print("----------------------------------------------------------")
    for t, br_estimator in enumerate(estimators):
        time = datetime.now()
        print("Estimator: {}".format(br_estimator.__class__.__name__))
        clf = GridSearchCV(br_estimator, tuned_parameters[t])
        clf.fit(X_train_fill, y_train)
        all_info[i] = clf.cv_results_
        grid_scores[i] = clf.best_score_
        print(grid_scores[i])
        best_params[i] = clf.best_params_
        print(best_params[i])
        i += 1
        print("Time: {}".format(datetime.now() - time))
        print("----------------------------------------------------------")


# good_imputer = IterativeImputer(random_state = 0, estimator = BayesianRidge())
# X_train_f = good_imputer.fit_transform(X_train)
# pd.DataFrame(X_train_f).to_csv("BayesianRidge_X_train.csv", index = False)

X_train_f = pd.read_csv("BayesianRidge_X_train.csv")

# вроде как лучшие
k_f = KFold(n_splits = 10, shuffle = True, random_state = 0)

# неожиданный и простой король -- эластичная сеть
algo_1 = ElasticNetCV(l1_ratio = 0.1, normalize = True, max_iter = 1500) # adaboost ломает его!
algo_1_score = cross_val_score(algo_1, X_train_f, y_train, scoring = 'r2', cv = k_f)
np.round(np.mean(algo_1_score), 3) # 0.942

# не неожиданный лес
algo_2 = RandomForestRegressor(n_estimators = 100, random_state = 0)
algo_2_score = cross_val_score(algo_2, X_train_f, y_train, scoring = 'r2', cv = k_f)
np.round(np.mean(algo_2_score), 3) # 0.917

# также не неожиданный бустинг
algo_3 = HistGradientBoostingRegressor(max_iter = 500, random_state = 0)
algo_3_score = cross_val_score(algo_3, X_train_f, y_train, scoring = 'r2', cv = k_f)
np.round(np.mean(algo_3_score), 3) # 0.9

# попробуем их всех объединить
estims = [
    ('RF', RandomForestRegressor(n_estimators = 100, random_state = 0)),
    ('GB', HistGradientBoostingRegressor(max_iter = 500, random_state = 0))
]

# лес и бустинг генерирует даннные дополнительно к исходным, на них обучаем эластичную сеть
# reg = StackingRegressor(estimators = estims, final_estimator = ElasticNetCV(l1_ratio = 0.1, normalize = True, max_iter = 1500), passthrough = False)
# reg_score = cross_val_score(reg, X_train_f, y_train, scoring = 'r2', cv = k_f)
# np.round(np.mean(reg_score), 3) # 0.918 -- не зашло

# лес и бустинг генерирует даннные, на них обучаем эластичную сеть
reg = StackingRegressor(estimators = estims, final_estimator = ElasticNetCV(l1_ratio = 0.1, normalize = True, max_iter = 1500), passthrough = True)
reg_score = cross_val_score(reg, X_train_f, y_train, scoring = 'r2', cv = k_f)
np.round(np.mean(reg_score), 3) # 0.952 -- пока что предел


# эластичная сеть и бустинг генерирует даннные дополнительно к исходным, на них обучаем лес

# estims = [
#     ('EN', ElasticNetCV(l1_ratio = 0.1, normalize = True, max_iter = 1500)),
#     ('GB', HistGradientBoostingRegressor(max_iter = 500, random_state = 0))
# ]

# reg = StackingRegressor(estimators = estims, final_estimator = RandomForestRegressor(n_estimators = 100, random_state = 0))
# reg_score = cross_val_score(reg, X_train_f, y_train, scoring = 'r2', cv = k_f)
# np.round(np.mean(reg_score), 3) # 0.927: хуже просто сети
