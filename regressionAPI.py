from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('trainedRecycleModel.pkl')

@app.route('/')
def index():
    html_template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Recycling Prediction</title>
    </head>
    <body>
        <h1>Enter Paper Information</h1>
        <form action="/predict" method="post">
            <label for="paperAmount">Paper Amount:</label>
            <input type="text" id="paperAmount" name="paperAmount" required><br><br>
            <input type="checkbox" id="woody" name="woody" value="1">
            <label for="woody">Woody</label><br>
            <input type="checkbox" id="newspaper" name="newspaper" value="1">
            <label for="newspaper">Newspaper</label><br>
            <input type="checkbox" id="mixed" name="mixed" value="1">
            <label for="mixed">Mixed</label><br>
            <input type="checkbox" id="inked" name="inked" value="1">
            <label for="inked">Inked</label><br>
            <input type="checkbox" id="cardboard" name="cardboard" value="1">
            <label for="cardboard">Cardboard</label><br><br>
            <input type="submit" value="Predict">
        </form>
    </body>
    </html>
    """
    return render_template_string(html_template)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the data from the request
        paperAmount = float(request.form['paperAmount'])
        woody = int(request.form.get('woody', 0))
        newspaper = int(request.form.get('newspaper', 0))
        mixed = int(request.form.get('mixed', 0))
        inked = int(request.form.get('inked', 0))
        cardboard = int(request.form.get('cardboard', 0))
        
        # Make prediction
        prediction = model.predict([[paperAmount, woody, newspaper, mixed, inked, cardboard]])
        
        # Prepare the response
        response = {'prediction': prediction[0]}
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)
