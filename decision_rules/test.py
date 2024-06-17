from scipy.io import arff
import pandas as pd
from survival.kaplan_meier_2 import KaplanMeierEstimator

df = pd.DataFrame(arff.loadarff(f"./GBSG2.arff")[0])
# code to change encoding of the file
tmp_df = df.select_dtypes([object])
tmp_df = tmp_df.stack().str.decode("utf-8").unstack()
for col in tmp_df:
    df[col] = tmp_df[col].replace({'?': None})


survival_time = df['survival_time'].to_numpy()
survival_status = df['survival_status'].to_numpy()

estimator = KaplanMeierEstimator()

estimator = estimator.fit(survival_time, survival_status)

print(estimator.surv_info_list)