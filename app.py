from flask import Flask, render_template, request
import joblib

# Load model and vectorizer
model = joblib.load('model/news_model.pkl')
vectorizer = joblib.load('model/vectorizer.pkl')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    news = request.form['news']
    transformed = vectorizer.transform([news])
    prediction = model.predict(transformed)[0]
    result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
    return render_template('index.html', prediction=result, news=news)

if __name__ == '__main__':
    app.run(debug=True)
