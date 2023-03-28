from math import sqrt
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

def evaluation(test_y, pred_test):
    MAE = mean_absolute_error(test_y, pred_test)
    MSE = mean_squared_error(test_y, pred_test)
    RMSE = sqrt(mean_squared_error(test_y, pred_test))
    R2 = r2_score(test_y, pred_test)
    return MAE, MSE, RMSE, R2