# CODE TO BE RUN IN KAGGLE NOTEBOOK

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# import dataset
df = pd.read_csv('../input/train.csv')

# analyse data
# missing data
df['Pclass'].isnull().value_counts()
df['Survived'].isnull().value_counts()

survivors = df.groupby('Pclass')['Survived'].agg(sum) # passengers survived in each class

# total passengers in each class
total_passengers = df.groupby('Pclass')['PassengerId'].count()
survivor_percentage = survivors / total_passengers

# plotting
fig = plt.figure()
ax = fig.add_subplot(111)
rect = ax.bar(survivors.index.values.tolist(), survivors, color='blue', width=0.5)
ax.set_ylabel('No. of survivors')
ax.set_title('Total number of survivors based on class')
xTickMarks = survivors.index.values.tolist()
ax.set_xticks(survivors.index.values.tolist())
xtickNames = ax.set_xticklabels(xTickMarks)
plt.setp(xtickNames, fontsize=20)
plt.show()

