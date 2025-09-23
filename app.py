from flask import Flask
from flask_cors import CORS
from routes import main

app = Flask(__name__)
CORS(app)


app.register_blueprint(main)

@app.route('/')
def home():
    
    return 'Hello World'

@app.route('/predict')
def predict():
    return 'Result'

if __name__ == '__main__':
    app.run()