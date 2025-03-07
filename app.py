from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load the saved models and encoders
ensemble_clf = joblib.load('ensemble_clf.pkl')
encoder = joblib.load('encoder.pkl')
label_encoder = joblib.load('label_encoder.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user input from the form
    persona = request.form.get('persona')
    usage = request.form.get('usage')
    processor = request.form.get('processor')
    ram = request.form.get('ram')
    graphics = request.form.get('graphics')
    storage = request.form.get('storage')
    display = request.form.get('display')
    battery = request.form.get('battery')

    # Create a DataFrame from user inputs
    new_user = pd.DataFrame({
        'Persona': [persona],
        'Usage': [usage],
        'Processor': [processor],
        'RAM': [ram],
        'Graphics': [graphics],
        'Storage': [storage],
        'Display': [display],
        'Battery Life': [battery]
    })

    # Encode the user input
    new_user_encoded = encoder.transform(new_user)

    # Predict the laptop specification label
    predicted_label = label_encoder.inverse_transform(ensemble_clf.predict(new_user_encoded))

    # Return the recommendation to the frontend
    return render_template('index.html', recommendation=predicted_label[0])

if __name__ == '__main__':
    app.run(debug=True)
