# InboxShield — SMS Spam Classifier

InboxShield is a Streamlit web app that classifies an SMS message as **Spam** or **Non-Spam** using a saved TF-IDF and Logistic Regression model. It also displays the model's estimated spam probability.

## Features

- Paste an SMS into the web app and analyze it interactively.
- See the predicted label and spam probability.
- Uses the same text preprocessing as the model training notebook.
- Responsive interface with animated visual elements.

## Project files

| File | Description |
| --- | --- |
| `app.py` | Streamlit application and text preprocessing |
| `sms_spam_logistic_regression.joblib` | Trained model pipeline and label mapping |
| `requirements.txt` | Python package dependencies |
| `E_Mail_classification_NLP.ipynb` | Notebook with the dataset exploration, training, and evaluation workflow |

The trained model file must be in the same directory as `app.py`.

## Requirements

- Python 3.10 or newer
- The model artifact: `sms_spam_logistic_regression.joblib`

## Setup on Windows

Open PowerShell in the project directory and create a virtual environment:

```powershell
python -m venv .venv
```

Activate it and install the dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell prevents script activation, you can use the environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Run the app

From the project directory, run:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Streamlit prints a local URL in the terminal, usually `http://localhost:8501`. Open that URL in a browser.

## How predictions work

1. The app lowercases the message.
2. It removes URLs, email addresses, and HTML tags, then normalizes whitespace.
3. The saved pipeline predicts the label and class probabilities.
4. The app displays the label and the probability assigned to the spam class.

The probability is a model estimate, not a guarantee. Treat unexpected links and requests for personal information with care, even if a message is classified as Non-Spam.

## Model compatibility

`requirements.txt` pins scikit-learn to `1.6.1`, the version used to serialize the included model. Keep this version aligned with the model artifact to avoid compatibility warnings or inference issues.

Only load `.joblib` model files from a trusted source. Joblib uses Python serialization and untrusted files can execute code when loaded.
