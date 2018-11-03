from fastai.structured import *
from sklearn import ensemble, preprocessing
from sklearn.model_selection import train_test_split

#to make that work we need do download and unzip data from https://www.kaggle.com/c/bluebook-for-bulldozers/data
df = pd.read_csv(f'bulldozers/Train.csv', low_memory=False, parse_dates=["saledate"])
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
    res = [rmse(m.predict(X_train), y_train),
           rmse(m.predict(X_valid), y_valid),
           m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


m = ensemble.RandomForestRegressor(n_jobs=-1)
m.fit(X_train, y_train)

test_X = pd.read_csv('bulldozers/Test.csv')
test_X = test_X.drop(columns=['MachineHoursCurrentMeter', 'auctioneerID'])

for feature in test_X.columns.values:
    if feature == 'auctioneerID' or feature == 'SalesID':
        continue
    test_X[feature].fillna('0', inplace=True)
    test_X[feature] = le.fit_transform(test_X[feature])

predicted_prices = m.predict(test_X)

my_submission = pd.DataFrame({'SalesID': test_X.SalesID, 'SalePrice': predicted_prices})
my_submission.to_csv('submission2.csv', index=False)
