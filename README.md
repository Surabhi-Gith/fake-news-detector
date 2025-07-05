# 📰 Fake News Detector

A machine learning-powered Flask web app that detects whether a news article is **Fake** or **Real** using NLP and a Passive Aggressive Classifier trained on the Fake/Real News dataset.

---

## 🚀 Features

- 🧠 Trained ML model using `TfidfVectorizer` + `PassiveAggressiveClassifier`
- 🌐 Flask-based frontend with live user input
- 🌗 Dark mode toggle for better UX
- 📊 100% accuracy (on this dataset)
- 📦 Clean file structure
- 🔥 Responsive, modern UI

---

## 🧪 Tech Stack

| Category       | Stack                        |
|----------------|------------------------------|
| Backend        | Python, Flask                |
| ML Libraries   | scikit-learn, Pandas, Joblib |
| Frontend       | HTML, CSS (Poppins Font)     |
| Deployment     | Render.com / GitHub Pages    |

---

## 🧰 Installation

```bash
# 1. Clone the repo
git clone https://github.com/Surabhi-Gith/fake-news-detector.git
cd fake-news-detector

# 2. Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run Flask App
python3 app.py

---

🧠 Model Training

The model was trained using notebooks/fake_news_model_training.py
It:
	•	Loads and combines real/fake news
	•	Vectorizes text with TF-IDF
	•	Trains PassiveAggressiveClassifier
	•	Saves model & vectorizer as .pkl

---

📂 Project Structure

fake-news-classifier/
├── app.py
├── templates/
│   └── index.html
├── data/
│   ├── Fake.csv
│   └── True.csv
├── model/
│   ├── news_model.pkl
│   └── vectorizer.pkl
├── notebooks/
│   └── fake_news_model_training.py
└── requirements.txt

---

📈 Dataset

	•	<a href="https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset" target="_blank">
🔗 Fake and Real News Dataset on Kaggle
</a>
	•	Combined two CSVs: Fake.csv and True.csv


