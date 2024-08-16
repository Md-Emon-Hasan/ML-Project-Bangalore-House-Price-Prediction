from flask import Flask
from flask import request
from flask import render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the model and data
pipe = pickle.load(open('models/pipe.pkl', 'rb'))
df = pickle.load(open('models/df.pkl', 'rb'))

@app.route('/', methods=['GET'])
def index():
    # Sort the unique values
    area_type_var = sorted(df['area_type'].unique())
    location_var = sorted(df['location'].unique())

    return render_template('index.html',
                           area_types=area_type_var,
                           locations=location_var)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve form data
    area_type = request.form['area_type']
    location = request.form['location']
    total_sqft = float(request.form['total_sqft'])
    bath = float(request.form['bath'])
    balcony = float(request.form['balcony'])
    bhk = int(request.form['bhk'])

    # Prepare query
    query = np.array([area_type, location, total_sqft, bath, balcony, bhk])
    query = query.reshape(1, 6)

    # Predict price
    predicted_price = int(np.exp(pipe.predict(query)[0]))

    # Sort the unique values again for consistent dropdown options
    area_type_sorted = sorted(df['area_type'].unique())
    location_sorted = sorted(df['location'].unique())

    # Render template with prediction and input values
    return render_template(
        'index.html',
        area_types=area_type_sorted,
        locations=location_sorted,
        price=predicted_price,
        area_type=area_type,
        location=location,
        total_sqft=total_sqft,
        bath=bath,
        balcony=balcony,
        bhk=bhk)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)