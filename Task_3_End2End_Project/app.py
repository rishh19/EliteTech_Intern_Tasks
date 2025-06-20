from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load the model
model = joblib.load("models/titanic_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            int(request.form['pclass']),
            int(request.form['sex']),
            float(request.form['age']),
            int(request.form['sibsp']),
            int(request.form['parch']),
            float(request.form['fare']),
            int(request.form['embarked']),
            int(request.form['alone'])
        ]
        prediction = model.predict([features])
        result = 'Survived' if prediction[0] == 1 else 'Did not survive'
        return render_template('index.html', prediction=result)
    except Exception as e:
        return render_template('index.html', prediction=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
