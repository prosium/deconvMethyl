import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear

deconv = 'bvls'

predicted_cell_proportions = np.full((props.shape[0], props.shape[1]), np.nan)


for i in range(props.shape[0]):
    result = lsq_linear(ref.values, df.loc[ref.index, df.columns[i]].values, bounds=(0, 1))
    predicted_cell_proportions[i, :] = result.x


predicted_df = pd.DataFrame(predicted_cell_proportions, columns=props.columns, index=props.index)


predicted_values = predicted_df.values.flatten()
true_values = props.values.flatten()
celltypes = np.tile(props.columns, props.shape[0])
methods = np.tile(method, predicted_df.shape[0] * predicted_df.shape[1])
deconv_methods = np.tile(deconv, predicted_df.shape[0] * predicted_df.shape[1])

result_df = pd.DataFrame({
    'Predicted value': predicted_values,
    'True value': true_values,
    'Celltype': celltypes,
    'Normalization method': methods,
    'Deconvolution method': deconv_methods
})

longlist = pd.concat([longlist, result_df], ignore_index=True)
