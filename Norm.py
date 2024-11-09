import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer

# 1. Z-Score Normalization
def z_score_normalization(data):
    """
    Z-Score Normalization: (x - mean) / std
    """
    return (data - np.nanmean(data)) / np.nanstd(data)

# 2. Min-Max Normalization
def min_max_normalization(data):
    """
    Min-Max Normalization: (x - min) / (max - min)
    """
    min_val = np.nanmin(data)
    max_val = np.nanmax(data)
    return (data - min_val) / (max_val - min_val)

# 3. Column-wise Z-Score Normalization
def col_z_score_normalization(data):
    """
    Column-wise Z-Score Normalization
    """
    return data.apply(lambda x: (x - np.nanmean(x)) / np.nanstd(x), axis=0)

# 4. Column-wise Min-Max Normalization
def col_min_max_normalization(data):
    """
    Column-wise Min-Max Normalization
    """
    return data.apply(lambda x: (x - np.nanmin(x)) / (np.nanmax(x) - np.nanmin(x)), axis=0)

# 5. Quantile Normalization
def quantile_normalization(data):
    """
    Quantile Normalization using sklearn's QuantileTransformer
    """
    qt = QuantileTransformer(output_distribution='uniform', random_state=0)
    # Null 값은 제거하고 정규화 수행 후 다시 반영
    non_na_data = data.dropna(axis=0, how='any')
    normalized_data = qt.fit_transform(non_na_data)
    result = pd.DataFrame(normalized_data, index=non_na_data.index, columns=non_na_data.columns)
    return result.reindex(index=data.index, columns=data.columns).fillna(data)

# 6. Log Normalization
def log_normalization(data):
    """
    Log Normalization: log(x), 음수나 0 값을 NA로 처리
    """
    log_data = np.log(data.replace(0, np.nan))
    log_data[np.isinf(log_data)] = np.nan  # 무한대 값을 NA로 대체
    return log_data.dropna()

