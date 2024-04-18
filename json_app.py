from flask import Flask, request, jsonify
import joblib

# Load the pre-trained model
model = joblib.load('model.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Ensure that the request is form data
    if not request.form:
        return jsonify({"error": "Missing form data in request"}), 400

    # Extract input data from the form
    sl = request.form.get('sl')
    sw = request.form.get('sw')
    pl = request.form.get('pl')
    pw = request.form.get('pw')

    # Check if any of the form fields are missing
    if not all([sl, sw, pl, pw]):
        return jsonify({"error": "Missing input data"}), 400

    try:
        # Convert input data to float
        sl = float(sl)
        sw = float(sw)
        pl = float(pl)
        pw = float(pw)
    except ValueError:
        return jsonify({"error": "Invalid input data"}), 400

    # Make prediction using the loaded model
    sample = np.array([sl, sw, pl, pw]).reshape(1, -1)
    prediction = model.predict(sample)
    pred = ['setosa', 'versicolor', 'virginica'][prediction[0]]

    # Return the prediction as JSON
    return jsonify({"prediction": pred})

if __name__ == '__main__':
    app.run(host='0.0.0.0')  # Set debug=False in a production environment
