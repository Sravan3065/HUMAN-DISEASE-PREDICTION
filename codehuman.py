from flask import Flask, request, render_template_string, redirect, url_for, flash, session
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from collections import Counter
from sklearn.metrics import accuracy_score

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load and prepare data
DATA_PATH = "DiseaseAndSymptoms1.csv"
data = pd.read_csv(DATA_PATH).dropna(axis=1)

# Prepare features and labels
X = data.iloc[:, 1:]  # Symptoms
y = data.iloc[:, 0]    # Disease

# Encode labels for the target
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y)

# One-Hot Encoding for categorical features
one_hot_encoder = OneHotEncoder(sparse_output=False)
X_encoded = one_hot_encoder.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=24)

# Train models
final_rf_model = RandomForestClassifier(random_state=18)
final_rf_model.fit(X_train, y_train)

final_nb_model = GaussianNB()
final_nb_model.fit(X_train, y_train)

final_svm_model = SVC()
final_svm_model.fit(X_train, y_train)

# Model evaluation
print("Model Evaluation:")
rf_accuracy = accuracy_score(y_test, final_rf_model.predict(X_test))
nb_accuracy = accuracy_score(y_test, final_nb_model.predict(X_test))
svm_accuracy = accuracy_score(y_test, final_svm_model.predict(X_test))

print(f"Random Forest Accuracy: {rf_accuracy:.2f}")
print(f"Naive Bayes Accuracy: {nb_accuracy:.2f}")
print(f"SVM Accuracy: {svm_accuracy:.2f}")

# Create symptom index for predictions
symptom_columns = one_hot_encoder.get_feature_names_out()
symptom_index = {symptom: index for index, symptom in enumerate(symptom_columns)}

def predict_disease(symptoms):
    input_data = [0] * len(symptom_index)
    invalid_symptoms = []

    for symptom in symptoms:
        symptom = symptom.strip().lower().replace('_', ' ')
        matched = False
        
        if symptom in symptom_index:
            input_data[symptom_index[symptom]] = 1
            matched = True
        else:
            for s in symptom_index.keys():
                if symptom in s:
                    input_data[symptom_index[s]] = 1
                    matched = True
                    break

        if not matched:
            invalid_symptoms.append(symptom)

    input_data = np.array(input_data).reshape(1, -1)

    predictions = {}
    if any(input_data[0]):
        rf_prediction_raw = final_rf_model.predict(input_data)
        nb_prediction_raw = final_nb_model.predict(input_data)
        svm_prediction_raw = final_svm_model.predict(input_data)

        rf_prediction = encoder.inverse_transform(rf_prediction_raw)[0]
        nb_prediction = encoder.inverse_transform(nb_prediction_raw)[0]
        svm_prediction = encoder.inverse_transform(svm_prediction_raw)[0]

        final_prediction = Counter([rf_prediction, nb_prediction, svm_prediction]).most_common(1)[0][0]

        predictions = {
            "Random Forest Prediction": rf_prediction,
            "Naive Bayes Prediction": nb_prediction,
            "SVM Prediction": svm_prediction,
            "Final Prediction": final_prediction
        }
    else:
        predictions = {"Error": "No valid symptoms provided.", "Invalid Symptoms": invalid_symptoms}

    return predictions

users = {}  # In-memory user storage

@app.route('/', methods=['GET'])
def home():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if user is registered
        if username in users and users[username] == password:
            session['username'] = username  # Store username in session
            flash('Login successful!', 'success')
            return redirect(url_for('index'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')

    login_html = """<!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Login</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #e9ecef; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; height: 100vh; }
            .container { background: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.2); width: 400px; text-align: center; }
            h1 { color: #343a40; margin-bottom: 20px; }
            label { font-weight: bold; display: block; margin-bottom: 5px; }
            input[type="text"], input[type="password"] { width: calc(100% - 22px); padding: 10px; margin-bottom: 15px; border: 1px solid #ced4da; border-radius: 4px; font-size: 14px; }
            input[type="submit"] { background: #28a745; color: white; border: none; padding: 10px 15px; cursor: pointer; border-radius: 5px; font-size: 16px; transition: background 0.3s; }
            input[type="submit"]:hover { background: #218838; }
            .register-link { margin-top: 10px; }
            .alert { padding: 10px; margin-bottom: 20px; }
            .alert-success { background-color: #d4edda; color: #155724; }
            .alert-danger { background-color: #f8d7da; color: #721c24; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Login</h1>
            {% with messages = get_flashed_messages(with_categories=true) %}
              {% if messages %}
                {% for category, message in messages %}
                  <div class="alert alert-{{ category }}">{{ message }}</div>
                {% endfor %}
              {% endif %}
            {% endwith %}
            <form method="POST">
                <label for="username">Username:</label>
                <input type="text" id="username" name="username" required>
                <label for="password">Password:</label>
                <input type="password" id="password" name="password" required>
                <input type="submit" value="Login">
            </form>
            <div class="register-link">
                <a href="/register">Not registered? Sign up here.</a>
            </div>
        </div>
    </body>
    </html>"""
    return render_template_string(login_html)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users:
            flash('Username already exists. Please choose another.', 'danger')
        else:
            users[username] = password  # Store user credentials
            flash('Registration successful! Please log in.', 'success')
            return redirect(url_for('login'))

    registration_html = """<!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Registration</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #e9ecef; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; height: 100vh; }
            .container { background: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.2); width: 400px; text-align: center; }
            h1 { color: #343a40; margin-bottom: 20px; }
            label { font-weight: bold; display: block; margin-bottom: 5px; }
            input[type="text"], input[type="password"] { width: calc(100% - 22px); padding: 10px; margin-bottom: 15px; border: 1px solid #ced4da; border-radius: 4px; font-size: 14px; }
            input[type="submit"] { background: #28a745; color: white; border: none; padding: 10px 15px; cursor: pointer; border-radius: 5px; font-size: 16px; transition: background 0.3s; }
            input[type="submit"]:hover { background: #218838; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Register</h1>
            {% with messages = get_flashed_messages(with_categories=true) %}
              {% if messages %}
                {% for category, message in messages %}
                  <div class="alert alert-{{ category }}">{{ message }}</div>
                {% endfor %}
              {% endif %}
            {% endwith %}
            <form method="POST">
                <label for="username">Username:</label>
                <input type="text" id="username" name="username" required>
                <label for="password">Password:</label>
                <input type="password" id="password" name="password" required>
                <input type="submit" value="Register">
            </form>
        </div>
    </body>
    </html>"""
    return render_template_string(registration_html)

@app.route('/index', methods=['GET', 'POST'])
def index():
    if 'username' not in session:
        flash('Please log in to access the prediction page.', 'warning')
        return redirect(url_for('login'))

    predictions = {}
    if request.method == 'POST':
        symptoms_input = [
            request.form['symptom1'],
            request.form['symptom2'],
            request.form['symptom3'],
            request.form['symptom4'],
            request.form['symptom5']
        ]
        predictions = predict_disease(symptoms_input)

    html_content = """<!doctype html>
    <html lang="en">
    <head>
        <meta charset="utf-8">
        <title>Disease Prediction</title>
        <style>
            body { font-family: Arial, sans-serif; background-color: #e9ecef; margin: 0; padding: 20px; display: flex; justify-content: center; align-items: center; height: 100vh; }
            .container { background: #ffffff; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0, 0, 0, 0.2); width: 400px; text-align: center; }
            h1 { color: #343a40; margin-bottom: 20px; }
            label { font-weight: bold; display: block; margin-bottom: 5px; }
            input[type="text"] { width: calc(100% - 22px); padding: 10px; margin-bottom: 15px; border: 1px solid #ced4da; border-radius: 4px; font-size: 14px; }
            input[type="submit"], input[type="button"] { background: #28a745; color: white; border: none; padding: 10px 15px; cursor: pointer; border-radius: 5px; font-size: 16px; transition: background 0.3s; }
            input[type="submit"]:hover, input[type="button"]:hover { background: #218838; }
            ul { list-style-type: none; padding: 0; margin-top: 20px; }
            li { background: #f8f9fa; padding: 10px; margin: 5px 0; border-radius: 4px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Disease Prediction</h1>
            <form method="POST">
                <label for="symptom1">Symptom 1:</label>
                <input type="text" id="symptom1" name="symptom1" required>
                <label for="symptom2">Symptom 2:</label>
                <input type="text" id="symptom2" name="symptom2" required>
                <label for="symptom3">Symptom 3:</label>
                <input type="text" id="symptom3" name="symptom3">
                <label for="symptom4">Symptom 4:</label>
                <input type="text" id="symptom4" name="symptom4">
                <label for="symptom5">Symptom 5:</label>
                <input type="text" id="symptom5" name="symptom5">
                <input type="submit" value="Predict">
                <input type="button" value="Reset" onclick="window.location.reload();">
            </form>
            <div>
                {% if predictions %}
                    <h2>Predictions:</h2>
                    <ul>
                        {% for key, value in predictions.items() %}
                            <li><strong>{{ key }}:</strong> {{ value }}</li>
                        {% endfor %}
                    </ul>
                {% endif %}
            </div>
        </div>
    </body>
    </html>"""
    return render_template_string(html_content, predictions=predictions)

if __name__ == '__main__':
    app.run(debug=True)
