import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold

years = [2022, 2021, 2020]

# Number of folds
n_folds = 3  # You can adjust this
# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)



# Initialize an empty list to store DataFrames
dfs = []

# Loop through each year, read the corresponding file and append to the list
for year in years:
    file_path = f'cleaned_datasets/Cleaned_Crimes_{year}.csv'  # Adjust the file path as needed
    df = pd.read_csv(file_path)
    dfs.append(df)

# Concatenate all DataFrames in the list into one
combined_df = pd.concat(dfs, ignore_index=True)

# Parse the date column to extract day, month, and year
combined_df['Date'] = pd.to_datetime(combined_df['Date'])
combined_df['Day'] = combined_df['Date'].dt.day
combined_df['Month'] = combined_df['Date'].dt.month
combined_df['Year'] = combined_df['Date'].dt.year

# Encoding categorical variables
le_location_desc = LabelEncoder()
combined_df['Location Description'] = le_location_desc.fit_transform(combined_df['Location Description'])

# Selecting relevant columns
features = ['Community Area', 'Location Description', 'Day', 'Month', 'Year']
target = 'Primary Type'

# Split the dataset
X = combined_df[features]
y = combined_df[target]

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Models to train
models = {
    "LogisticRegression": LogisticRegression(),
    #"DecisionTree": DecisionTreeClassifier(),
    #"RandomForest": RandomForestClassifier(),
    #"MultinomialNB": MultinomialNB(),
    #"KNN": KNeighborsClassifier(),
    #"SVM": SVC(),
    #"MLPClassifier": MLPClassifier()
}

# Dictionary to store the scores
scores = {name: [] for name in models.keys()}

# Stratified K-Fold Cross-Validation
for train_index, test_index in skf.split(X_scaled, y):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    for name, model in models.items():
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        score = accuracy_score(y_test, predictions)
        scores[name].append(score)

# Average scores across folds
average_scores = {name: sum(score) / len(score) for name, score in scores.items()}