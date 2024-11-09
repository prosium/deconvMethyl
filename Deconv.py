import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear, nnls
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from statsmodels.regression.linear_model import OLS
from sklearn.preprocessing import normalize
import logging

logging.basicConfig(level=logging.INFO)

# 1. BVLS
def bvls_deconvolution(ref, df):
    predictions = []
    for i in range(df.shape[1]):
        res = lsq_linear(ref, df.iloc[:, i], bounds=(0, 1))
        predictions.append(res.x)
    return np.array(predictions)

# 2. DCQ
def dcq_deconvolution(ref, df):
    predictions = []
    for i in range(df.shape[1]):
        coef = np.linalg.lstsq(ref, df.iloc[:, i], rcond=None)[0]
        predictions.append(np.clip(coef, 0, 1))  # 음수 값을 0으로, 최대 1로 제한
    return np.array(predictions)

# 3. ElasticNet
def elastic_net_deconvolution(ref, df):
    model = ElasticNet(alpha=0.5, l1_ratio=0.5)
    predictions = []
    for i in range(df.shape[1]):
        model.fit(ref, df.iloc[:, i])
        predictions.append(np.clip(model.coef_, 0, 1))
    return np.array(predictions)

# 4. EMeth-Binom
def emeth_binom_deconvolution(ref, df, max_iter=50):
    predictions = []
    for i in range(df.shape[1]):
        eta = np.zeros(ref.shape[1])
        rho = np.linalg.lstsq(ref, df.iloc[:, i], rcond=None)[0]
        for _ in range(max_iter):
            rho = np.clip(rho, 0, 1)
            eta = np.dot(ref.T, df.iloc[:, i] - np.dot(ref, rho))
            rho += eta
        predictions.append(rho)
    return np.array(predictions)

# 5. EMeth-Laplace
def emeth_laplace_deconvolution(ref, df, max_iter=50):
    predictions = []
    for i in range(df.shape[1]):
        eta = np.zeros(ref.shape[1])
        rho = np.linalg.lstsq(ref, df.iloc[:, i], rcond=None)[0]
        for _ in range(max_iter):
            rho = np.clip(rho, 0, 1)
            eta = np.median(df.iloc[:, i] - np.dot(ref, rho))
            rho += eta
        predictions.append(rho)
    return np.array(predictions)

# 6. EMeth-Normal
def emeth_normal_deconvolution(ref, df, max_iter=50):
    predictions = []
    for i in range(df.shape[1]):
        eta = np.zeros(ref.shape[1])
        rho = np.linalg.lstsq(ref, df.iloc[:, i], rcond=None)[0]
        for _ in range(max_iter):
            rho = np.clip(rho, 0, 1)
            eta = np.mean(df.iloc[:, i] - np.dot(ref, rho))
            rho += eta
        predictions.append(rho)
    return np.array(predictions)

# 7. EpiDISH
def epidish_deconvolution(ref, df):
    predictions = np.linalg.lstsq(ref, df.values, rcond=None)[0].T
    return predictions

# 8. FARDEEP
def fardeep_deconvolution(ref, df):
    predictions = []
    for i in range(df.shape[1]):
        coef = np.linalg.lstsq(ref, df.iloc[:, i], rcond=None)[0]
        coef[coef < 0] = 0
        coef /= coef.sum()  # 정규화
        predictions.append(coef)
    return np.array(predictions)

# 9. Lasso
def lasso_deconvolution(ref, df):
    model = Lasso(alpha=1)
    predictions = []
    for i in range(df.shape[1]):
        model.fit(ref, df.iloc[:, i])
        predictions.append(np.clip(model.coef_, 0, 1))
    return np.array(predictions)

# 10. Meth_atlas
def meth_atlas_deconvolution(ref, df):
    predictions = np.linalg.lstsq(ref, df.values, rcond=None)[0].T
    return predictions

# 11. MethylResolver
def methylresolver_deconvolution(ref, df):
    predictions = []
    for i in range(df.shape[1]):
        coef = np.linalg.lstsq(ref, df.iloc[:, i], rcond=None)[0]
        coef = np.clip(coef, 0, 1)
        predictions.append(coef)
    return np.array(predictions)

# 12. Minfi
def minfi_deconvolution(ref, df):
    predictions = np.linalg.lstsq(ref, df.values, rcond=None)[0].T
    return predictions

# 13. NNLS
def nnls_deconvolution(ref, df):
    predictions = []
    for i in range(df.shape[1]):
        x, _ = nnls(ref, df.iloc[:, i])
        predictions.append(x)
    return np.array(predictions)

# 14. OLS
def ols_deconvolution(ref, df):
    predictions = []
    for i in range(df.shape[1]):
        model = OLS(df.iloc[:, i], ref).fit()
        coef = model.params
        coef[coef < 0] = 0
        predictions.append(coef / coef.sum())
    return np.array(predictions)

# 15. Ridge
def ridge_deconvolution(ref, df):
    model = Ridge(alpha=0)
    predictions = []
    for i in range(df.shape[1]):
        model.fit(ref, df.iloc[:, i])
        predictions.append(np.clip(model.coef_, 0, 1))
    return np.array(predictions)

# 16. ElasticNet
def elastic_net_deconvolution(ref, df):
    model = ElasticNet(alpha=0.5, l1_ratio=0.5)
    predictions = []
    for i in range(df.shape[1]):
        model.fit(ref, df.iloc[:, i])
        predictions.append(np.clip(model.coef_, 0, 1))
    return np.array(predictions)
