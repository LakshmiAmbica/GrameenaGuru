from flask import Flask, render_template, request, session
from ibm_watsonx_ai.foundation_models import ModelInference
from deep_translator import GoogleTranslator

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a strong secret key

# 🌐 Language mapping
LANGUAGE_CODES = {
    "en": "english",
    "te": "telugu",
    "hi": "hindi"
}

# 🔐 WatsonX Model Setup
model = ModelInference(
    model_id="ibm/granite-3-8b-instruct",
    project_id="8992bfd7-0da8-4a68-abc5-a6a3853c661a",  # Your IBM project ID
    credentials={
        "apikey": "heElJmvLgP2kvXmdWoczCepZ8rsnSw_VZCndyI91ZFp5",  # Your API key
        "url": "https://eu-gb.ml.cloud.ibm.com"
    }
)

@app.route('/')
def index():
    session.setdefault("chat_history", [])
    return render_template('index.html', history=session["chat_history"])

@app.route('/ask', methods=['POST'])
def ask():
    user_question = request.form['question']
    selected_language = request.form['language']

    # Translate input to English if needed
    if selected_language != 'en':
        translated_input = GoogleTranslator(source=LANGUAGE_CODES[selected_language], target="english").translate(user_question)
    else:
        translated_input = user_question

    # Generate response using IBM model
    response = model.generate(
        prompt=translated_input,
        params={"max_new_tokens": 300, "temperature": 0.7}
    )
    english_output = response['results'][0]['generated_text']

    # Translate answer back to selected language
    if selected_language != 'en':
        translated_output = GoogleTranslator(source="english", target=LANGUAGE_CODES[selected_language]).translate(english_output)
    else:
        translated_output = english_output

    # Save to history
    session["chat_history"].append({"user": user_question, "ai": translated_output})
    session.modified = True

    return render_template('index.html', response=translated_output, history=session["chat_history"])

if __name__ == '__main__':
    app.run(debug=True)
