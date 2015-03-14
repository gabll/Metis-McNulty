import matplotlib.pyplot as plt
import pandas as pd
import pymysql
import numpy as np
import model_selection_utils
import credentials

db = pymysql.connect(host=credentials.ip,
                     user=credentials.user,
                     passwd=credentials.password,
                     db='bank')

df = pd.read_sql("select * from bank_additional_2", db)
db.close()

#Binaryze outcome var
df['y'][df['y'] == 'yes'] = 1
df['y'][df['y'] == 'no'] = 0

features_list = list(df.columns)
features_list.remove('y')
features_list.remove('duration') #See dataset description

outcome_var = 'y'
features_list.append(outcome_var)
    
df_conv = model_selection_utils.convert_features(df[features_list], outcome_var, dummies=True,
                                                 scaling=False, only_important_features=True, null_value='unknown')

X = df_conv.drop([outcome_var], 1)
Y = df_conv[outcome_var]
features_list_dummies = X.columns

X.columns = [i.replace('.', '') for i in X.columns]
X.columns = [i.replace('-', '_') for i in X.columns]

from sklearn.cross_validation import train_test_split

#Shuffle X and Y (necessary for plotting the learning curve)
X, _, Y, _ = train_test_split(X,Y, test_size=0, random_state=1)

#Split test and training set
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=.20, random_state=42)

from sklearn.linear_model import LogisticRegression

PREDICTOR = LogisticRegression().fit(X, Y)

import flask

# Initialize the app
app = flask.Flask(__name__)

# Homepage
@app.route("/")
def viz_page():
    """ Homepage: serve our visualization page, awesome.html
    """
    with open("bank-dss-mvp.html", 'r') as viz_file:
        return viz_file.read()

# Get an example and return it's score from the predictor model
@app.route("/pred", methods=["POST"])
def pred():
    """  When A POST request with json data is made to this uri,
         Read the example from the json, predict probability and
         send it with a response
    """
    # Get decision score for our example that came with the request
    data = flask.request.json
    print data    
    x = np.matrix(data["example"])
    print x    
    score = PREDICTOR.predict_proba(x)
    # Put the result in a nice dict so we can send it as json
    print score    
    results = {"pred": list(score[:,1])}
    print results    
    return flask.jsonify(results)

#--------- RUN WEB APP SERVER ------------#

# Start the app server on port 80
# (The default website port)

app.run(host='0.0.0.0', port=8000)
