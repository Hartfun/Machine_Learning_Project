from flask import Flask, request, render_template, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models and scaler
l_reg = joblib.load('logreg_model.pkl')
mlp = joblib.load('mlp_model.pkl')
rf = joblib.load('rf_model.pkl')
svm = joblib.load('svm_model.pkl')
svm_poly = joblib.load('svm_poly_model.pkl')
scaler = joblib.load('scaler.pkl')

# Define preprocessing
def preprocess_input(form):
    input_dict = {
        # Example fields; update field names to match your df columns (after dummies)
        'field1': [int(form['field1'])],
        'field2': [int(form['field2'])],
        # Add all required fields
    }
    X_new = pd.DataFrame(input_dict)
    # One-hot encode, align with training dummies
    # X_new = pd.get_dummies(X_new, columns=[...], drop_first=True)
    X_new_scaled = scaler.transform(X_new)
    return X_new_scaled

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        data = request.form
        X_new_scaled = preprocess_input(data)
        prediction = rf.predict(X_new_scaled)
        # You can let user choose the model (l_reg, mlp, etc.)
        result = {
            'Would_you_recommend_your_university_to_others': int(prediction[0][0]),
            'Are_you_satisfied_with_the_university\'s_facilities': int(prediction[0][1]),
            'Do_you_prefer_online_classes_over_offline_classes': int(prediction[0][2]),
            'Do_you_live_in_a_hostel': int(prediction[0][3])
        }
        return render_template('result.html', result=result)
    return render_template('form.html')

# Or add an API endpoint
@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.get_json(force=True)
    X_new_scaled = preprocess_input(data)
    prediction = rf.predict(X_new_scaled)
    output = prediction[0].tolist()
    return jsonify({'prediction': output})

if __name__ == "__main__":
    app.run(debug=True)
