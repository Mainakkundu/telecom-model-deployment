from flask import Flask, jsonify, request
#from preprocessing.functions import tokenize
import xgboost as xgb
import joblib
from healthcheck import HealthCheck

import os
import logging
import flask

logging.basicConfig(format='%(message)s', level=logging.INFO)
app = Flask(__name__)

#target={0:'Debt collection', 1:'Credit card or prepaid card', 2:'Mortgage', 
        #3:'Checking or savings account', 4:'Student loan', 
        #5:'Vehicle loan or lease'}

#tfvectorizer = joblib.load('models/tfvectroizer.pkl') 
#xgb_clf = xgb.Booster({'nthread': 3})
model = joblib.load("./model.joblib")

logging.info('All models loaded succcessfully')

health = HealthCheck(app, "/hcheck")



COL_NAMES = ['customerID', 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling','PaymentMethod', 'MonthlyCharges', 'TotalCharges']
#instance = ['7317-GGVPB', 'Male', 0, 'Yes', 'No', 71, 'Yes', 'Yes', 'Fiber optic', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'Two year', 'Yes', 'Credit card (automatic)', 108.6, '7690.9']





def howami():
    return True, "I am alive. Thanks for checking.."

health.add_check(howami)





    
    
@app.route('/score', methods=['POST'])
def predict_fn():
    features = request.get_json()['text']
    input_feature = pd.DataFrame([features],columns=COL_NAMES)
    logging.info('Received incoming message - '+ input_feature)
    predictions = model.predict(input_feature)
    return jsonify({'predictions ': str(predictions)})

@app.route('/')
def hello():
    return 'Welcome to Churn Prediction'

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))