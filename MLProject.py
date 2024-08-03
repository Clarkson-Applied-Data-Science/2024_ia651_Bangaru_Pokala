# %% [markdown]
# # Machine Learning Project
# ## Gold Stock Price Prediction

# %%
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.preprocessing import LabelEncoder

# %%
d1= pd.read_csv('FINAL_USO.csv')
print(d1.head(10))

# %%
data_types = d1.dtypes
statistics = d1.describe()
missing_values = d1.isnull().sum()
print(missing_values, statistics, data_types)

# %% [markdown]
# No Missing Values

# %% [markdown]
# Some useful plots to visualize information (EDA):

# %% [markdown]
# Histograms

# %%
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.figure(figsize=(20, 15))

key_features = ['Open', 'High', 'Low', 'Close', 'Volume', 'SP_open', 'SP_high', 'SP_low', 'SP_close', 'SP_volume', 'GDX_Open', 'GDX_High', 'GDX_Low', 'GDX_Close', 'GDX_Volume', 'USO_Open', 'USO_High', 'USO_Low', 'USO_Close', 'USO_Volume']
d1[key_features].hist(bins=30, figsize=(20, 15), layout=(7, 3))
plt.tight_layout()
plt.show()


# %%
#Trends Over Time

d1= pd.read_csv('FINAL_USO.csv')
d1['Date'] = pd.to_datetime(d1['Date'])

d1.set_index('Date', inplace=True)

plt.figure(figsize=(15, 10))
plt.plot(d1.index, d1['USO_Close'], label='USO_Close')
plt.plot(d1.index, d1['SP_close'], label='SP_close')
plt.plot(d1.index, d1['GDX_Close'], label='GDX_Close')
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.title('Trends of USO, S&P 500, and GDX Closing Prices Over Time')
plt.legend()
plt.show()

# %% [markdown]
# Relationships between variables

# %%
# Scatter plot for 'USO_Close' vs 'SP_close'
plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
sns.scatterplot(x=d1['SP_close'], y=d1['USO_Close'])
plt.xlabel('S&P 500 Close')
plt.ylabel('USO Close')
plt.title('USO Close vs S&P 500 Close')

# Scatter plot for 'USO_Close' vs 'GDX_Close'
plt.subplot(1, 2, 2)
sns.scatterplot(x=d1['GDX_Close'], y=d1['USO_Close'])
plt.xlabel('GDX Close')
plt.ylabel('USO Close')
plt.title('USO Close vs GDX Close')

plt.tight_layout()
plt.show()

# %% [markdown]
# Feature Engineering

# %%
import pandas as pd

d1= pd.read_csv('FINAL_USO.csv')

d1['Date'] = pd.to_datetime(d1['Date'])

# Creating lags for 'Open', 'High', and 'Low'
high_low_open_columns = [col for col in d1.columns if 'Open' in col or 'High' in col or 'Low' in col]

lag_intervals = [1, 2, 3] 

for column in high_low_open_columns:
    for lag in lag_intervals:
        d1[f'{column}_lag_{lag}'] = d1[column].shift(lag)

d1.dropna(inplace=True)

target_columns = [col for col in d1.columns if 'Close' in col]

X = d1[[col for col in d1.columns if 'lag' in col]]
y = d1[target_columns]

print("Features (X):", X.head())


# %%
print("\nTarget Variables (y):", y)

# %%
y= y.drop(columns= ['GDX_Adj Close', 'Adj Close', 'USO_Adj Close'])
print("\nTarget Variables (y):", y)

# %% [markdown]
# Feature Importance with Random Forest

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

base_regressor = RandomForestRegressor(n_estimators=10, random_state=42)

# MultiOutputRegressor for multi-target prediction
multi_target_regressor = MultiOutputRegressor(base_regressor)

multi_target_regressor.fit(X_train, y_train)

importances = sum(est.feature_importances_ for est in multi_target_regressor.estimators_) / len(multi_target_regressor.estimators_)

feature_importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Extracting important features with importance greater than 0.01
important_features = feature_importance_df[feature_importance_df['Importance'] > 0.01]['Feature'].tolist()

print("Important Features (Importance > 0.01):")
print(important_features)


# %% [markdown]
# Correlation Matrix

# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

specified_features = [
    'GDX_Low_lag_1', 'Low_lag_1', 'USO_Open_lag_3', 'GDX_Open_lag_2', 'GDX_High_lag_1', 
    'USO_High_lag_3', 'GDX_High_lag_3', 'Open_lag_1', 'High_lag_1', 
    'USO_High_lag_1', 'USO_Low_lag_2', 'USO_Open_lag_1', 'GDX_High_lag_2', 'GDX_Open_lag_3', 
    'Low_lag_2', 'High_lag_3', 'USO_Low_lag_1'
]

target_columns = [col for col in d1.columns if 'Close' in col]
y = d1[target_columns]

# Adjusted Close is unwanted
y = y.drop(columns=['GDX_Adj Close', 'Adj Close', 'USO_Adj Close'])

relevant_columns = specified_features + y.columns.tolist()

correlation_matrix = d1[relevant_columns].corr()

plt.figure(figsize=(16, 14))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='coolwarm', 
    linewidths=0.5, 
    fmt=".2f",
    cbar_kws={"shrink": .8}, 
    vmin=-1, vmax=1
)

plt.title('Correlation Matrix of Selected Features and Closing Prices', fontsize=20)
plt.xticks(rotation=45, ha='right', fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout() 
plt.show()


# %% [markdown]
# Data Splitting and Scaling

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_scaled[:5], X_test_scaled[:5]

# %% [markdown]
# Initial Model Training
# Linear Regression

# %%
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

model = LinearRegression()
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)

mse = mean_squared_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print(f'MSE: {mse}, RMSE: {rmse}, R-squared: {r2}')

# %% [markdown]
# Logistic Regression

# %%
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

median_uso_close = d1['USO_Close'].median()
d1['USO_Close_Binary'] = (d1['USO_Close'] > median_uso_close).astype(int)
y_binary = d1['USO_Close_Binary']

X_train_bin, X_test_bin, y_train_bin, y_test_bin = train_test_split(X, y_binary, test_size=0.2, random_state=42)

logistic_regressor = LogisticRegression(max_iter=1000)
logistic_regressor.fit(X_train_bin, y_train_bin)

y_pred_bin = logistic_regressor.predict(X_test_bin)
accuracy = accuracy_score(y_test_bin, y_pred_bin)
conf_matrix = confusion_matrix(y_test_bin, y_pred_bin)
class_report = classification_report(y_test_bin, y_pred_bin)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# %% [markdown]
# PCA

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

d1['Date'] = pd.to_datetime(d1['Date'])

specified_features = [
    'GDX_Low_lag_1', 'Low_lag_1', 'USO_Open_lag_3', 'GDX_Open_lag_2', 'GDX_High_lag_1', 
    'USO_High_lag_3', 'GDX_High_lag_3', 'Open_lag_1', 'High_lag_1', 
    'USO_High_lag_1', 'USO_Low_lag_2', 'USO_Open_lag_1', 'GDX_High_lag_2', 'GDX_Open_lag_3', 
    'Low_lag_2', 'High_lag_3', 'USO_Low_lag_1'
]

X_pca = d1[specified_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_pca)

pca = PCA()
X_pca_transformed = pca.fit_transform(X_scaled)

explained_variance = pca.explained_variance_ratio_

plt.figure(figsize=(8, 6))
plt.bar(range(1, len(explained_variance) + 1), explained_variance, alpha=0.5, align='center', label='Individual explained variance')
plt.step(range(1, len(explained_variance) + 1), np.cumsum(explained_variance), where='mid', label='Cumulative explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.title('Explained Variance by Principal Components')
plt.show()

# Scree Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.grid(True)
plt.show()

principal_df = pd.DataFrame(X_pca_transformed[:, :2], columns=['Principal Component 1', 'Principal Component 2'])

plt.figure(figsize=(10, 8))
plt.scatter(principal_df['Principal Component 1'], principal_df['Principal Component 2'], alpha=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D PCA of the Dataset')
plt.grid(True)
plt.show()


# %% [markdown]
# Disclosing which features are contributing more to principal components

# %%
loadings = pca.components_

loadings_df = pd.DataFrame(loadings.T, columns=[f'PC{i+1}' for i in range(loadings.shape[0])], index=specified_features)

print("Feature Contributions to Principal Components:")
print(loadings_df)

# %% [markdown]
# Support Vector Machine
# for Regression

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

d1['Date'] = pd.to_datetime(d1['Date'])

specified_features = [
    'GDX_Low_lag_1', 'Low_lag_1', 'USO_Open_lag_3', 'GDX_Open_lag_2', 'GDX_High_lag_1', 
    'USO_High_lag_3', 'GDX_High_lag_3', 'Open_lag_1', 'High_lag_1', 
    'USO_High_lag_1', 'USO_Low_lag_2', 'USO_Open_lag_1', 'GDX_High_lag_2', 'GDX_Open_lag_3', 
    'Low_lag_2', 'High_lag_3', 'USO_Low_lag_1'
]

X = d1[specified_features]

target_columns = [col for col in d1.columns if 'Close' in col]
y = d1[target_columns]

y = y.drop(columns=['GDX_Adj Close', 'Adj Close', 'USO_Adj Close'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svr_models = {}
for target in y.columns:
    svr = SVR(kernel='rbf')
    svr.fit(X_train_scaled, y_train[target])
    svr_models[target] = svr

for target in y.columns:
    y_pred = svr_models[target].predict(X_test_scaled)
    mae = mean_absolute_error(y_test[target], y_pred)
    mse = mean_squared_error(y_test[target], y_pred)
    print(f"Evaluation for {target}:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}\n")


# %% [markdown]
# Random Forest

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

d1['Date'] = pd.to_datetime(d1['Date'])

specified_features = [
    'GDX_Low_lag_1', 'Low_lag_1', 'USO_Open_lag_3', 'GDX_Open_lag_2', 'GDX_High_lag_1', 
    'USO_High_lag_3', 'GDX_High_lag_3', 'Open_lag_1', 'High_lag_1', 
    'USO_High_lag_1', 'USO_Low_lag_2', 'USO_Open_lag_1', 'GDX_High_lag_2', 'GDX_Open_lag_3', 
    'Low_lag_2', 'High_lag_3', 'USO_Low_lag_1'
]

X = d1[specified_features]

target_columns = [col for col in d1.columns if 'Close' in col]
y = d1[target_columns]

y = y.drop(columns=['GDX_Adj Close', 'Adj Close', 'USO_Adj Close'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_models = {}
for target in y.columns:
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train[target])
    rf_models[target] = rf

for target in y.columns:
    y_pred = rf_models[target].predict(X_test)
    mae = mean_absolute_error(y_test[target], y_pred)
    mse = mean_squared_error(y_test[target], y_pred)
    print(f"Evaluation for {target}:")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}\n")

# Printing feature importances
feature_importances = rf_models['Close'].feature_importances_
importance_df = pd.DataFrame({'Feature': specified_features, 'Importance': feature_importances})
importance_df = importance_df.sort_values(by='Importance', ascending=False)
print("Feature Importances for 'Close':")
print(importance_df)


# %% [markdown]
# Bagging

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

bagging_models = {}
for target in y.columns:
    bagging = BaggingRegressor(n_estimators=100, random_state=42)
    bagging.fit(X_train, y_train[target])
    bagging_models[target] = bagging

for target in y.columns:
    y_pred = bagging_models[target].predict(X_test)
    mae = mean_absolute_error(y_test[target], y_pred)
    mse = mean_squared_error(y_test[target], y_pred)
    print(f"Evaluation for {target} (Bagging):")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}\n")


# %% [markdown]
# Boosting

# %%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

boosting_models = {}
for target in y.columns:
    boosting = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
    boosting.fit(X_train, y_train[target])
    boosting_models[target] = boosting

for target in y.columns:
    y_pred = boosting_models[target].predict(X_test)
    mae = mean_absolute_error(y_test[target], y_pred)
    mse = mean_squared_error(y_test[target], y_pred)
    print(f"Evaluation for {target} (Boosting):")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"Mean Squared Error: {mse:.4f}\n")


