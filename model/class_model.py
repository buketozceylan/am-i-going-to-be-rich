import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("main-data.csv")

cat_col = ["Experience Level", "Employment Type","Job Title"]

ordinal_encoder = OrdinalEncoder(categories = [["SE","MI","EN","EX"],["FT","PT","CT","FL"],["Data Scientist","Data Engineer","Data Analyst","Machine Learning Engineer"]])

preprocessor = ColumnTransformer(transformers = [
    ("transformation_name",ordinal_encoder,cat_col)
],remainder = "passthrough")


remote = [0,1,2]
for i in df["Remote?"].unique():
    if i == 0:
        df["Remote?"] = df["Remote?"].replace(i, 0)
    elif i == 50:
        df["Remote?"] = df["Remote?"].replace(i, 1)
    elif i == 100:
        df["Remote?"] = df["Remote?"].replace(i, 2)
        

X = df.drop("Is rich?", axis = 1)
y = df["Is rich?"]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=15)
X_cols = X_train.columns
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

scaler = RobustScaler()
cols = preprocessor.get_feature_names_out()
X_train = pd.DataFrame(X_train_transformed, columns = cols, index = X_train.index)
X_test = pd.DataFrame(X_test_transformed, columns = cols, index = X_test.index)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
X_train = pd.DataFrame(X_train, columns = X_cols)
X_test = pd.DataFrame(X_test, columns = X_cols)
X_train = X_train.drop("Unnamed: 0", axis =1)
X_test = X_test.drop("Unnamed: 0", axis =1)

rfc = RandomForestClassifier(n_estimators = 500, random_state=15, min_samples_split= 2,max_features=None,max_depth=None)
rfc.fit(X_train, y_train)

y_pred = rfc.predict(X_test)
acc_score = accuracy_score(y_test,y_pred)
class_report = classification_report(y_test,y_pred)
conf_matrix = confusion_matrix(y_test,y_pred)