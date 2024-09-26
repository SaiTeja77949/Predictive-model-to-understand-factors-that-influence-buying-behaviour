# Predictive-model-to-understand-factors-that-influence-buying-behaviour
Predictive Model using RandomForestClassifier

This project aims to develop a machine learning model to predict the likelihood of a customer completing a booking based on various features. The dataset used for this project is customer_booking.csv, which contains information about customer bookings, including sales channel, trip type, flight day, route, booking origin, number of passengers, purchase lead time, length of stay, flight hour, and flight duration.

Data Preprocessing
The dataset is first loaded into a Pandas dataframe using pd.read_csv. The encoding parameter is set to "ISO-8859-1" to ensure proper encoding of the data. The head() method is used to display the first few rows of the dataframe, and the info() method is used to display summary statistics about the dataframe.

The route column is encoded using one-hot encoding to convert categorical variables into numerical variables. This is done using pd.get_dummies. The resulting dataframe is stored in df_encoded.

The numerical columns num_passengers, purchase_lead, length_of_stay, flight_hour, and flight_duration are scaled using StandardScaler to have zero mean and unit variance.

Feature Engineering
The target variable is booking_complete, which indicates whether the customer completed the booking or not. The features are all the other columns in the dataframe, excluding booking_complete.

Model Training and Evaluation
The dataset is split into training and testing sets using train_test_split with a test size of 0.2 and a random state of 42. A random forest classifier is trained on the training data using RandomForestClassifier with 500 estimators and a random state of 42.

The model is evaluated using accuracy score and classification report. The accuracy score is calculated using accuracy_score, and the classification report is generated using classification_report.

Feature Importance
The feature importances are obtained from the random forest model using feature_importances_. A dataframe is created with the feature names and importances, and sorted in descending order of importance. The top 10 most important features are visualized using a bar plot with seaborn.

Results
The accuracy of the model is printed to the console, along with the classification report. The feature importances are visualized in a bar plot, showing the top 10 most important features.

Dependencies
This project requires the following dependencies:

pandas
numpy
scikit-learn
matplotlib
seaborn
