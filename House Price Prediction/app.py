from flask import Flask, render_template, request
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib

app = Flask(__name__)

# Assuming you have a trained model (best_model) for house price prediction
# This is just a placeholder; in a real scenario, you'd load your trained model.

best_model = joblib.load(r'C:\Users\syedk\Downloads\Bharat-Intern\House Price Prediction\best_model.pkl')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        features = [float(x) for x in request.form.values()]
        prediction = predict_price(features)
        return render_template('result.html', prediction=prediction)
    return render_template('index.html')

def predict_price(features):
    # Here, you'll need to preprocess the features (e.g., convert to numpy array)
    # and use your best_model to predict the price.
    # For this example, I'm just returning a placeholder value.
    # prediction = best_model.predict([features])
    prediction = np.random.randint(100000, 500000)  # Placeholder prediction
    return prediction

if __name__ == '__main__':
    app.run(debug=True)
