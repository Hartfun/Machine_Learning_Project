# Machine_Learning_Project

1. Data Preparation and Pre-processing
Missing Values: Checked with df.isnull().sum().

Encoding: 'Yes'/'No' responses converted to 1/0.

One-hot Encoding: Applied to gender and UG/PG status columns.

Correlation Matrix: Generated and visualized with seaborn heatmap.

Target Variables: Defined as the four questions above.

Feature/Target Split: Predictors (X) and targets (y) separated.

Train/Test Split: 70/30 split.

Imbalance Visualization: Plotted skewness for each encoded feature.

Standardization: Applied using StandardScaler.

2. Model Training and Evaluation
The following models are trained and evaluated for multi-label classification using the MultiOutputClassifier wrapper:

a. Logistic Regression
Trained and tested with max 1,000 iterations.

Prints confusion matrix and classification report for each label.

b. Multi-Layer Perceptron Classifier
Network structure: 3 hidden layers with 100 neurons each.

Prints metrics for each target.

c. Random Forest Classifier
100 estimators, fixed random seed.

Prints classification results by target.

d. Support Vector Machine (Linear and Polynomial Kernels)
Both linear and polynomial (poly) kernels tested.

Provides detailed evaluation for each target variable.

3. Model Comparison by Accuracy
For each model, the average accuracy across the four target variables is computed.

Results are collected and displayed as a summary DataFrame, sorted by average accuracy percentage.

Expected Key Outcomes
Correlation Heatmap: Visualizes relationships among all encoded features and targets; helps identify feature dependencies or redundant columns.

Imbalance Bar Plot: Shows which features are imbalanced, which can indicate model bias or the need for further pre-processing.

Per-Model Evaluation: For each model, confusion matrices and classification reports (with precision, recall, F1-score, and support) give a detailed view of model strengths/weaknesses across the targets.

Model Ranking: The final table ranks the models by their average prediction accuracy (across targets), quickly showing which algorithm performed best for your multi-label classification task on this survey dataset.
