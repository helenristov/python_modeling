<h1 align="center">Python Script Using Scikit-Learn for Development and Evaluation of Different Machine Learning Models </h1>

**I will cover the following:**

* [Data Loading](#data-loading)
* [Data Imputations](#data-impute)
* [Data Plotting and Exploratory Data Analysis ](#data-plots)
* [Standardizing Data](#data-standardization)
* [Data Partitions For Models](#data-partition)
* [Model Evaluation](#model-evaluation)



### Data Loading
```python
# Loading dataset
url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
data = pd.read_csv(url)
display(data)  ## --> This is a prettier version that print to view dataframes
target_var = 'Survived'
type_of_model = 'Boolean'

X = data.drop(columns=[target_var])
y = data[target_var]

```
### Data Imputations
```python
### Check to Identify which columns have null values
missing_values = data.isnull().sum()

missing_columns = missing_values[missing_values > 0].index.tolist()
print(f"Columns with missing values: {', '.join(missing_columns)}")

##impute missing values
# Identify numerical and categorical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Impute numerical features with median
numerical_imputer = SimpleImputer(strategy='median')
X[numerical_features] = numerical_imputer.fit_transform(X[numerical_features])

# Impute categorical features with the most frequent value
categorical_imputer = SimpleImputer(strategy='most_frequent')
X[categorical_features] = categorical_imputer.fit_transform(X[categorical_features])

# Check for remaining missing values after imputation
missing_values_after = X.isnull().sum()
na_values_after = X.isna().sum()
print(f"Missing values after imputation:\n{missing_values_after}")
print(f"\nNaN values after imputation:\n{missing_values_after}")
# Display the first few rows of the imputed dataset
print("\nImputed dataset:")
display(X)

```

### Data Plotting and Exploratory Data Analysis
```python
#plot distributions
 # Plot distributions of numerical features
numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
for feature in numerical_features:
    plt.figure(figsize=(10, 6))
    sns.histplot(X[feature], kde=True, bins=30)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency')
    plt.show()

# Plot distributions of categorical features
categorical_features = X.select_dtypes(include=['object']).columns
for feature in categorical_features:
    plt.figure(figsize=(10, 6))
    sns.countplot(x=X[feature])
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.show()

```
### Standardizing Data
```python
##drop any fields like name and ticket number that are unique identifiers
X = X.drop(columns=['Name', 'Ticket'])
print(X.head())

##standardizing the data for the model

numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Preprocessing for numerical data
numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Apply the preprocessing
X_preprocessed = preprocessor.fit_transform(X)

# Convert the result to a DataFrame for easy viewing (optional)
X_preprocessed_df = pd.DataFrame(X_preprocessed.toarray() if hasattr(X_preprocessed, "toarray") else X_preprocessed)

# Display the preprocessed features
print("\nPreprocessed features:\n", X_preprocessed_df.head())

```
### Data Partitions
```python
# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_preprocessed_df, y, test_size=0.2, random_state=42)

print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

```

### Model Evaluation
```python
## model evaluation
#Initialize the models
models = {
    "Logistic Regression": LogisticRegression(max_iter=200),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Support Vector Machine": SVC(),
    "Naive Bayes": GaussianNB()
}

# Function to evaluate a model
def evaluate_model(model, X_train, y_train, X_test, y_test):
    ## TODO:  Should incorporate hyper-tuning here
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

# Compare models
results = {}
for model_name, model in models.items():
    accuracy = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[model_name] = accuracy
    print(f"{model_name}: {accuracy:.2f}")

# Show comparison results
results_df = pd.DataFrame(list(results.items()), columns=['Model', 'Accuracy'])
print("\nModel Comparison:\n", results_df.sort_values('Accuracy', ascending=False))

```
