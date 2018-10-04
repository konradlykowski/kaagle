from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display
from sklearn import preprocessing

from sklearn import metrics

from sklearn import metrics


# df_raw = pd.read_csv(f'bulldozers/Train.csv', low_memory=False, parse_dates=["saledate"])
# pd.to_pickle(df_raw, "bulldozers/Train.pkl")

def show_correlation_matrix(corr_matrix):
    plt.figure(figsize=(10, 10))
    plt.imshow(corr_matrix, cmap='RdYlGn', interpolation='none', aspect='auto')
    plt.colorbar()
    plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation='vertical')
    plt.yticks(range(len(corr_matrix)), corr_matrix.columns);
    plt.suptitle('Correlations Heat Map', fontsize=15, fontweight='bold')
    plt.show()


def show_violin_plot(data_frame, feature):
    fig, ax = plt.subplots()
    ax.violinplot(data_frame[feature], vert=False)
    plt.suptitle(feature, fontsize=15, fontweight='bold')
    plt.show()


def display_all(df):
    with pd.option_context("display.max_rows", 1000, "display.max_columns", 1000):
        display(df)


df = pd.read_pickle("bulldozers/Train.pkl")
le = preprocessing.LabelEncoder()

df['SalePrice'] = np.log(df['SalePrice'])

df = df.drop(columns=['MachineHoursCurrentMeter', 'auctioneerID'])

for feature in df.columns.values:

    if feature == 'auctioneerID':
        continue
    df[feature].fillna('0', inplace=True)
    df[feature] = le.fit_transform(df[feature])

y = df.pop('SalePrice')
X = df

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.33, random_state=42)


def rmse(x, y): return math.sqrt(((x - y) ** 2).mean())


def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
           m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


print("CZESC")
print(df.isnull().values.any())
print("CZESC")

m = RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)
print_score(m)
