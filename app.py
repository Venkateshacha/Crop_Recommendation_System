from flask import Flask, request, render_template
import pandas as pd
import pickle

app = Flask(__name__)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.preprocessing import OrdinalEncoder,LabelEncoder,OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
df = pd.read_csv("C:\\Users\\Venkatesh\\Downloads\\Crop_recommendation.csv")
#TARGET COLUMN
class_labels = df['label'].unique().tolist()
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])
class_labels = le.classes_
#SPLIT THE DATA
x = df.drop('label',axis=1)
y = df['label']
features_data = {'columns': list(x.columns)}
#Train the model
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,shuffle=True)

#BUILD MODEL
rf_model = RandomForestClassifier()
rf_model.fit(x_train,y_train)
y_pred = rf_model.predict(x_test)

#HYPER PARAMETER TUNING
rf = RandomForestClassifier()
param_grid = {'n_estimators':np.arange(50,200),
    'criterion':['gini','entropy'],
    'max_depth':np.arange(2,25),
    'min_samples_split':np.arange(2,25),
    'min_samples_leaf':np.arange(2,25)}
rscv_model = RandomizedSearchCV(rf,param_grid, cv=5)
rscv_model.fit(x_train,y_train)
rscv_model.best_estimator_

#MODEL EVALUATION TEST
new_rf_model = rscv_model.best_estimator_
y_pred = new_rf_model.predict(x_test)
y_pred_train = new_rf_model.predict(x_train)
features_data = {'columns':list(x.columns)}
test_series = pd.Series(np.zeros(len(features_data['columns'])),index=features_data['columns'])

import pickle
with open('new_rf_model.pickle','wb') as file:
    pickle.dump(new_rf_model, file)

# Load the model
with open('new_rf_model.pickle', 'rb') as file:
    model = pickle.load(file)
# Load the column names
with open('features_data.pickle', 'rb') as file:
    features_data = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/recommendation')
def recommendation():
    return render_template('recommendation.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Load the model and features_data as before
    # Get the input values from the form
    N = float(request.form['N'])
    P = float(request.form['P'])
    K = float(request.form['K'])
    temperature = float(request.form['temperature'])
    humidity = float(request.form['humidity'])
    ph = float(request.form['ph'])
    rainfall = float(request.form['rainfall'])
    # Create a DataFrame from the input values
    input_df = pd.DataFrame([[N, P, K, temperature, humidity, ph, rainfall]], columns=features_data['columns'])
    # Make prediction
    crop_index = model.predict(input_df)[0]
    recommended_crop = class_labels[crop_index]
    return render_template('recommendation.html', recommended_crop= recommended_crop)

if __name__ == '__main__':
    app.run(debug=True)
