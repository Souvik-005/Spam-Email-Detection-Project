import re
import numpy as np
import joblib
from flask import Flask, request, render_template

app = Flask(__name__)
model = joblib.load('Model/Email.joblib')

FEATURES = [
    'word_freq_make', 'word_freq_address', 'word_freq_all', 'word_freq_3d', 
    'word_freq_our', 'word_freq_over', 'word_freq_remove', 'word_freq_internet', 
    'word_freq_order', 'word_freq_mail', 'word_freq_receive', 'word_freq_will', 
    'word_freq_people', 'word_freq_report', 'word_freq_addresses', 'word_freq_free', 
    'word_freq_business', 'word_freq_email', 'word_freq_you', 'word_freq_credit', 
    'word_freq_your', 'word_freq_font', 'word_freq_000', 'word_freq_money', 
    'word_freq_hp', 'word_freq_hpl', 'word_freq_george', 'word_freq_650', 
    'word_freq_lab', 'word_freq_labs', 'word_freq_telnet', 'word_freq_857', 
    'word_freq_data', 'word_freq_415', 'word_freq_85', 'word_freq_technology', 
    'word_freq_1999', 'word_freq_parts', 'word_freq_pm', 'word_freq_direct', 
    'word_freq_cs', 'word_freq_meeting', 'word_freq_original', 'word_freq_project', 
    'word_freq_re', 'word_freq_edu', 'word_freq_table', 'word_freq_conference', 
    'char_freq_;', 'char_freq_(', 'char_freq_[', 'char_freq_!', 'char_freq_$', 
    'char_freq_#', 'capital_run_length_average', 'capital_run_length_longest', 
    'capital_run_length_total'
]

def extract_features(text):
    words = re.findall(r'\w+', text.lower())
    total_words = len(words) if len(words) > 0 else 1
    total_chars = len(text) if len(text) > 0 else 1
    
    cap_runs = re.findall(r'[A-Z]+', text)
    if cap_runs:
        run_lengths = [len(run) for run in cap_runs]
        cap_avg = sum(run_lengths) / len(run_lengths)
        cap_max = max(run_lengths)
        cap_total = sum(run_lengths)
    else:
        cap_avg, cap_max, cap_total = 0, 0, 0

    feature_values = []
    
    for feat in FEATURES[:48]:
        target = feat.replace('word_freq_', '')
        count = words.count(target)
        feature_values.append((count / total_words) * 100)
        
    chars_to_check = [';', '(', '[', '!', '$', '#']
    for char in chars_to_check:
        count = text.count(char)
        feature_values.append((count / total_chars) * 100)
        
    feature_values.extend([cap_avg, float(cap_max), float(cap_total)])
    
    return np.array(feature_values).reshape(1, -1)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    email_text = request.form.get('email_text', '')
    if not email_text:
        return render_template('index.html', prediction="Empty Text")
    
    numeric_input = extract_features(email_text)
    
    pred = model.predict(numeric_input)[0]
    result = "✖ SPAM" if pred == 1 else "✓ NOT SPAM"
    
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)