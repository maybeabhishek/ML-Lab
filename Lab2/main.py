from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM',
                'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
data = pd.read_csv('/home/maybeabhishek/Documents/Projects/ML-Lab/Datasets/boston-house-prices/housing.csv',
                   header=None, delimiter=r"\s+", names=column_names)
print(data.head(5))

# Dimension of the dataset
print(np.shape(data))

# Dimension of the dataset
print(np.shape(data))

#Histogram
data.hist(figsize=(15,10),grid=False)
plt.show()

# outlier percentage
for k, v in data.items():
    q1 = v.quantile(0.25)
    q3 = v.quantile(0.75)
    irq = q3 - q1
    v_col = v[(v <= q1 - 1.5 * irq) | (v >= q3 + 1.5 * irq)]
    perc = np.shape(v_col)[0] * 100.0 / np.shape(data)[0]
    print("Column %s outliers = %.2f%%" % (k, perc))

#correlation Heat Map
plt.figure(figsize=(20, 10))
sns.heatmap(data.corr().abs(),  annot=True)
plt.show()

