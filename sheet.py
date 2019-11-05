# Preprocessing
# https://towardsdatascience.com/data-preprocessing-in-python-b52b652e37d5

## Pandas
df.head()
df.columns
df.info()
df.describe()
df.isnull()
df.apply(<lambda/function>)
df.fillna(value) of df.fillna(method=method)
method = {‘backfill’, ‘bfill’, ‘pad’, ‘ffill’, None}

df.concat([col1, col2])
df.append(df1)

df.drop([list of columns], axis=1, inplace=True)
df.iloc[].values
dataset['column'].replace({'No': 0, 'Yes': 1},inplace = True)
dataset = pd.get_dummies(dataset, columns=categorical_columns)

pd.cut(x, bins)
pd.qcut(x, q)


## Pyplot / Seaborn
matplotlib.rcParams['figure.figsize']=(5, 5)


## Sklearn
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean') 
imputer = imputer.fit(X[:, 1:])
X[:, 1:] = imputer.transform(X[:, 1:])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 0] = labelencoder_X.fit_transform(X[:, 0])
onehotencoder = OneHotEncoder(categorical_features=[0])
X = onehotencoder.fit_transform(X).toarray()
labelencoder_Y = LabelEncoder()
Y = labelencoder_Y.fit_transform(Y)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


from scipy import stats
z = np.abs(stats.zscore(dataset._get_numeric_data()))
# print(dataset)
dataset= dataset[(z < 3).all(axis=1)]
dataset.shape

from sklearn.model_selection import cross_val_score


from sklearn.model_selection import cross_val_predict
y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score, f1_score


from sklearn.decomposition import PCA
pca = PCA(n_components = 2)
X2D = pca.fit_transform(X)

from sklearn.decomposition import KernelPCA
rbf_pca = KernelPCA(n_components = 2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
X_train.shape, X_test.shape


# Machine Learning Models
## Sklearn

### DT
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(
    criterion='gini',
    max_depth=10,
    min_samples_split=10,
    min_samples_leaf=10,
    max_features=0.7,
    random_state=42
)

from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
dot_data = export_graphviz(
    classifier, out_file=None, 
    feature_names=X.columns,  
    filled=True, rounded=True,  
    special_characters=True
)  

graph = graphviz.Source(dot_data)  
graph

### Kmeans
imageSamples = shuffle(imageArray, random_state=42)[:1000]
classifier = KMeans(n_clusters=nColors, random_state=42)

from scipy.cluster.hierarchy import dendrogram, linkage
import scipy.cluster.hierarchy as shc

plt.figure(figsize=(10, 7))
plt.title("Iris Dendograms")
dend = shc.dendrogram(shc.linkage(X_norm, method='ward'))

from sklearn.cluster import AgglomerativeClustering
# Out of the 4 linkage: single, complete, ward and average. Ward performs the best
model = AgglomerativeClustering(n_clusters=3, linkage='ward')
model.fit(pca_features)
labels = model.labels_

sns.scatterplot(pca_features[:, 0], pca_features[:, 1], hue=labels, palette=sns.color_palette("muted")[:3])
plt.show()

from sklearn.cluster import SpectralClustering
model = SpectralClustering(n_clusters=3)
model.fit(pca_features)
labels = model.labels_

sns.scatterplot(pca_features[:, 0], pca_features[:, 1], hue=labels, palette=sns.color_palette("muted")[:3])
plt.show()

### SVM
from sklearn import svm
svm.SVC
svm.LinearSVC
svm.SVR

### Linear
from sklearn.linear_model import LinearRegression, Lasso, Rigde, ElasticNet, SGDClassifier, LogisticRegression

### Polynomial
from sklearn.preprocessing import PolynomialFeatures 
  
poly = PolynomialFeatures(degree = 4) 
X_poly = poly.fit_transform(X) 
- Then apply linear regression