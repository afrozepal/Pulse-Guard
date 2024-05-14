import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer

##import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import sklearn as sk
import seaborn as sns
# from google.colab import files
from sklearn.model_selection import train_test_split

##install pyreadstat module
# !pip install pyreadstat

#Acquiring The Data



## Loading the Data

# uploaded = files.upload()
df = pd.read_csv('Heart_Disease_Prediction.csv')
print(df.shape)
df.describe()

### to see what dataset actually looks like
df.head()

##dimensions
df.shape

###to check datatypes of columns -tell about features and their type of data
df.info()

##basic descriptive to see , count , mean , std , max

pd.set_option('display.float_format' , lambda x: '%.3f' % x)
df.describe().transpose()

# Filtering Data

## Checking for null entries, infinite values, duplicates

# finding missing values
df.isnull()

## which columns have missing values and to check the extent of missing data

df.isnull().sum()/len(df)*100     ## /len(df)*100 to check percentage respective to columns count , without it gives just count of null values

### check duplicates
df.duplicated().sum()

df.info()

## Encoding categiorical features

## ttransforming text to integers

df['Heart Disease'].unique()

df['Heart Disease'].value_counts(normalize=True)

df['Heart Disease']=df['Heart Disease'].replace(['Presence' ,'Absence'],[1,0])  ##if heart disease its 1 , else 0

df['Heart Disease'].unique()

#### to get dummies
df.info()

df.shape

df.describe()

# Exploring Data

## Data Visualization

### Numerical Features

#### Box Plots

# Outliers
# We have calculated first quartile (q1), third quartile (q3), and interquartile range (iqr) for every current feature using the quantile method. Then, we computed the lower and upper bounds for detecting outliers based on the IQR.
# We have stored Outliers in a dictionary having multiple lists, each having multiple values that basically corresponds to outlier values and list name correspond to specific feature of dataset that has outlier values.
# In our dataset:
# Among 14 features, outliers exist in 7 features named as:
# 1.	Outliers in Chest pain type: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# 2.	Outliers in BP: [174, 178, 180, 200, 192, 178, 180, 180, 172]
# 3.	Outliers in Cholesterol: [564, 407, 417, 409, 394]
# 4.	Outliers in FBS over 120: [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
# 5.	Outliers in Max HR: [71]
# 6.	Outliers in ST depression: [4.2, 5.6, 4.2, 6.2]
# 7.	Outliers in Number of vessels fluro: [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
# Strategies:
# •	Here different strategies can be adopted. As in chest pain, all the outliers values are same so its possible that they are not true outliers but some category so we can leave them. Same goes for vessels fluro and FBS over 120.
# •	However in case of BP all values are different, here capping (setting /choosing a certain threshold to accommodate maximum and minimum values in a certain range) can be used to handle outliers. Same goes for cholesterol, ST depression
# •	Outlier in max HR is only one value and is not suitable for maximum heart rate. so we can choose to discard it.


import matplotlib.pyplot as plt

# Define a list of numerical features
numerical_features = ['Age', 'BP', 'Cholesterol', 'Max HR', 'ST depression']

# Create a boxplot for each feature
plt.figure(figsize=(12, 8))
for i, feature in enumerate(numerical_features, 1):
    plt.subplot(3, 3, i)
    plt.boxplot(df[feature])
    plt.title(feature)
plt.tight_layout()
plt.show()
# Print outliers for each feature
# yers in {col}: {outlier_vals}")


print(df)

#### Histograms

# Determine the number of rows and columns dynamically
num_plots = len(numerical_features)
cols = 2
rows = int(np.ceil(num_plots / cols))

fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

for i, feature in enumerate(numerical_features):
    row = i // cols
    col = i % cols
    axes[row, col].hist(df[feature], bins=20, alpha=0.7)
    axes[row, col].set_title(feature)
    axes[row, col].set_xlabel('Value')
    axes[row, col].set_ylabel('Frequency')

# If the number of plots is odd, remove the last subplot if it's empty
if num_plots % 2 != 0 and len(numerical_features) > 1:
    axes[rows-1, cols-1].axis('off')

plt.tight_layout()
plt.show()


# BP: Almost symmetrical distribution; not skewed.

# Sex, EKG, Thallium, Excercise Angina: Significant Skewness, because skewness measures the asymmetry of the distribution of a variable. However, for binary categorical variables like "Sex, EKG, Thallium, Excercise Angina" which only have two categories, the concept of skewness doesn't apply in the same way as it does for continuous numerical variables.

# Max HR: Negatively Skewed.

# Slope of ST, Number of vessels fluro: Highly Positively Skewed. Data concentrated on the right side of the distribution. => not numerical

# ST Depression: The data points seem to cluster around the median and extend to the upper quartile without any significant deviation towards lower values. => categorical




#### Scatter Plots



import matplotlib.pyplot as plt
import seaborn as sns

numerical_features = df.select_dtypes(include=['int64', 'float64'])

subplot_height = 4
subplot_width = 4

num_features = len(numerical_features.columns)
num_cols = min(num_features, 4)
num_rows = (num_features - 1) // num_cols + 1

fig, axes = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(subplot_width * num_cols, subplot_height * num_rows))

if num_rows == 1:
    axes = [axes]
if num_cols == 1:
    axes = [[ax] for ax in axes]

for i, col1 in enumerate(numerical_features.columns):
    for j, col2 in enumerate(numerical_features.columns):
        if i != j and i < j:
            row_idx = (i * num_cols + j - 1) // num_cols
            col_idx = (i * num_cols + j - 1) % num_cols
            if row_idx < num_rows:
                axes[row_idx][col_idx].scatter(df[col1], df[col2], alpha=0.5)
                axes[row_idx][col_idx].set_title(f'{col1} vs {col2}')
                axes[row_idx][col_idx].set_xlabel(col1)
                axes[row_idx][col_idx].set_ylabel(col2)

plt.tight_layout()
plt.show()


### Categorical Features

#### Bar Graphs

categorical_features = ['FBS over 120', 'Sex', 'Slope of ST', 'Thallium', 'Chest pain type', 'EKG results', 'Exercise angina', 'Number of vessels fluro', 'Heart Disease']

# Determine the number of rows and columns dynamically
num_plots = len(categorical_features)
cols = 2
rows = int(np.ceil(num_plots / cols))

fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
fig.subplots_adjust(hspace=0.5, wspace=0.5)

for i, feature in enumerate(categorical_features):
    row = i // cols
    col = i % cols
    counts = df[feature].value_counts()
    counts.plot(kind='bar', ax=axes[row, col], color='skyblue')
    axes[row, col].set_title(feature)
    axes[row, col].set_xlabel('Category')
    axes[row, col].set_ylabel('Count')

# If the number of plots is odd, remove the last subplot if it's empty
if num_plots % 2 != 0 and len(categorical_features) > 1:
    axes[rows-1, cols-1].axis('off')

plt.tight_layout()
plt.show()

# *Absence of heart disease is more common than its presence in the given dataset.*

### All Features

import seaborn as sns

sns.pairplot(df)
plt.show()


df['Heart Disease']=df['Heart Disease'].replace(['Presence' ,'Absence'],[1,0])  ##if heart disease its 1 , else 0
df['Heart Disease'] = df['Heart Disease'].astype(int)


## Testing & Training


df.info()

df.info
X = df.drop(columns=['Heart Disease'])
X.shape
print(X)

Y = df['Heart Disease']                   #separating array into input x and output y components
Y.shape
#print(y)

X_train ,X_test, Y_train, Y_test = train_test_split(X ,Y, test_size = 0.3 , random_state= 0)

from sklearn.preprocessing import StandardScaler
sc_train = StandardScaler().fit(X_train)
X_train_sc = sc_train.transform(X_train)

np.set_printoptions(precision=3)        #snapshot of transformed data
print(X_train_sc[0:5:])

#create an instance of linear regression logistic model for binary classification of heart disease
# if it exist or not like yes or no
from sklearn.linear_model import LogisticRegression
model_lr = LogisticRegression(solver = 'liblinear')

## using train data  to train the model features
## using train data  to train the model features

model_lr.fit(X_train_sc,Y_train)

##testing the model
#scaling test features  for optimal model performance
from sklearn.preprocessing import StandardScaler
sc_test = StandardScaler().fit(X_test)
X_test_sc  = sc_test.transform(X_test)

#TEST the performance of model on test data
res = model_lr.score(X_test_sc, Y_test)
res*100

#TEST the performance of model on train data
res = model_lr.score(X_train_sc, Y_train)
res*100

df.info()

df['Heart Disease'] = df['Heart Disease'].astype(int)

df.info()

# Drop the 'Heart Disease' column from the DataFrame
df_without_target = df.drop(columns=['Heart Disease'])

# Calculate correlations of each feature with the target variable
correlation_with_target = df_without_target.corrwith(df['Heart Disease'])

# Plot the correlation coefficients
c = ['red','green','yellow','blue','purple','orange','cyan']
correlation_with_target.plot.bar(figsize=(20, 10), title='Correlation with Heart Disease', fontsize=15, rot=45, grid=True, color=c)


# Cross Validation


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

## generate instancces of kfold logistics regression algorithm
kfold=KFold(n_splits=10,random_state=7,shuffle=True)
lr_2=LogisticRegression(solver='liblinear')

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)

##train the logistic regression Model w/Kfold cross validationn and get the scores
results= cross_val_score (lr_2 ,X_train_sc ,Y_train,cv=kfold)
results

 ## generate average accuracy
results.mean()*100,results.std()*100

## Logistic Regression with Log Loss Metric

# Logiistic Reression and CV with log loss metric
# we can use other scpring else than defualt scoring
kfold=KFold(n_splits=10 , random_state=7 ,shuffle=True)
lr_3=LogisticRegression(solver='liblinear')
results= cross_val_score (lr_3 ,X_train_sc ,Y_train,cv=kfold , scoring='neg_log_loss')
results.mean() , results.std()

## Logistic Regression with AUc metric

#Logistic regression with AUc metric
# not a good to get score
kfold=KFold(n_splits=10 , random_state=7 ,shuffle=True)
lr_4=LogisticRegression(solver='liblinear')
results= cross_val_score (lr_4 ,X_train_sc ,Y_train,cv=kfold , scoring='roc_auc')
results.mean() , results.std()

## Confusion Matrix

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

pred_y = model_lr.predict(X_test_sc)
matrix= confusion_matrix(Y_test , pred_y)
sns.heatmap(matrix/np.sum(matrix) , fmt='.2%' ,annot=True)

## classification REport
## it gives precision , recall , f1 score and support
## along with accuracy , macro and weighted avg

report= classification_report(Y_test,pred_y)
print(report)


### List of Features
print(df)
features = list(df_without_target)
features

### listt of coefficients
coefficients = model_lr.coef_
coefficients

## list of coeff from model . convert to list
coefficients = coefficients.ravel().tolist()
coefficients

df = {"Features": features, "Coefficients": coefficients}

# Create a DataFrame
coeff_table = pd.DataFrame(df)

# Sort the DataFrame by 'Coefficients'
coeff_table = coeff_table.sort_values(by='Coefficients')

# Display the sorted DataFrame
print(coeff_table)

coeff_table.plot(kind='bar', figsize=(20, 10))
plt.xticks(np.arange(len(features)), features)
plt.show()

## MAE, MSE

from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
# Calculate Mean Absolute Error (MAE)
mae = mean_absolute_error(Y_test, pred_y)

# Calculate Root Mean Squared Error (RMSE)
rmse = np.sqrt(mean_squared_error(Y_test, pred_y))

print("Mean Absolute Error (MAE):", mae)
print("Root Mean Squared Error (RMSE):", rmse)

## Silhouette Score

from sklearn.metrics import silhouette_score

# Assuming you have performed clustering and obtained cluster labels
# For example, if you used KMeans clustering:
from sklearn.cluster import KMeans

# Define the KMeans model with a specified number of clusters
kmeans = KMeans(n_clusters=3)

# Fit the model to your data and obtain cluster labels
cluster_labels = kmeans.fit_predict(X_train_sc)  # X is your data

# Calculate silhouette score
silhouette_avg = silhouette_score(X_train_sc, cluster_labels)

print("The average silhouette score is:", silhouette_avg)

# Save the trained model
with open('heart_disease_model.pkl', 'wb') as file:
    pickle.dump(model_lr, file)

