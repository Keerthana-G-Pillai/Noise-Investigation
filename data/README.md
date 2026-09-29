# Data

## HTRU2 Pulsar Classification Dataset

The project uses the HTRU2 dataset from the UCI Machine Learning Repository.

| Property | Value |
|---|---:|
| Observations | 17,898 |
| Features | 8 continuous numerical features |
| Target | Binary pulsar/non-pulsar classification |
| Non-pulsars | 16,259 (90.84%) |
| Pulsars | 1,639 (9.16%) |

### Features

1. Profile_mean
2. Profile_stdev
3. Profile_skewness
4. Profile_kurtosis
5. DM_mean
6. DM_stdev
7. DM_skewness
8. DM_kurtosis

### Dataset policy

The dataset is not stored directly in this repository. The final notebook contains the dataset-loading workflow used by the project.

### Why HTRU2 was selected

The dataset is suitable because its predictors are numerical, controlled feature perturbation can be applied consistently, it is a real binary classification task, and its strong class imbalance makes minority-class effects of label noise measurable.

Run the final notebook from a fresh environment to obtain the dataset through the project's loading workflow.

