import os
import pickle
import numpy as np
from flask import Flask, request, render_template_string

app = Flask(__name__)

# -------------------------------------------------------------------------
# 1. Load the trained Pickle Model
# -------------------------------------------------------------------------
MODEL_PATH = 'tree_pkl.pkl'
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
else:
    # Fallback mock model in case the pickle file isn't in the same directory yet
    class MockModel:
        def predict(self, X): return ['Alive']
        def predict_proba(self, X): return [[0.85, 0.15]]
    model = MockModel()

# -------------------------------------------------------------------------
# 2. Combined HTML/CSS Frontend Template
# -------------------------------------------------------------------------
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cancer Prognosis Predictor</title>
    <script src="https://cdn.jsdelivr.net/npm/@tailwindcss/browser@4"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body { font-family: 'Inter', sans-serif; }
    </style>
</head>
<body class="bg-slate-50 min-h-screen flex flex-col justify-between">

    <header class="bg-white border-b border-slate-200 py-5 px-6 shadow-xs">
        <div class="max-w-6xl mx-auto flex items-center justify-between">
            <div class="flex items-center space-x-3">
                <div class="bg-indigo-600 text-white p-2 rounded-xl shadow-md">
                    <svg class="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z"></path></svg>
                </div>
                <div>
                    <h1 class="text-xl font-bold text-slate-800 tracking-tight">OncoPredict Analytics</h1>
                    <p class="text-xs text-slate-500">Decision Tree Classifier System</p>
                </div>
            </div>
        </div>
    </header>

    <main class="max-w-4xl w-full mx-auto p-4 md:p-8 flex-grow">
        
        {% if prediction_text %}
        <div class="mb-8 p-6 rounded-2xl border transition-all duration-300 shadow-sm
            {% if prediction_class == 'alive' %} bg-emerald-50 border-emerald-200 text-emerald-900 {% else %} bg-rose-50 border-rose-200 text-rose-900 {% endif %}">
            <div class="flex items-center space-x-4">
                <div class="p-3 rounded-xl {% if prediction_class == 'alive' %} bg-emerald-500 text-white {% else %} bg-rose-500 text-white {% endif %}">
                    {% if prediction_class == 'alive' %}
                    <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4.318 6.318a4.5 4.5 0 000 6.364L12 20.364l7.682-7.682a4.5 4.5 0 00-6.364-6.364L12 7.636l-1.318-1.318a4.5 4.5 0 00-6.364 0z"></path></svg>
                    {% else %}
                    <svg class="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"></path></svg>
                    {% endif %}
                </div>
                <div>
                    <h2 class="text-2xl font-bold">{{ prediction_text }}</h2>
                    {% if confidence %}
                    <p class="text-sm opacity-80 mt-0.5">Model Confidence Assessment: <span class="font-semibold">{{ confidence }}%</span></p>
                    {% endif %}
                </div>
            </div>
        </div>
        {% endif %}

        {% if error_text %}
        <div class="mb-8 p-4 bg-amber-50 border border-amber-200 text-amber-900 rounded-2xl flex items-center space-x-2">
            <span class="font-medium">⚠️ {{ error_text }}</span>
        </div>
        {% endif %}

        <div class="bg-white rounded-2xl shadow-xl border border-slate-100 overflow-hidden">
            <div class="bg-indigo-900 px-8 py-6 text-white bg-gradient-to-r from-indigo-900 to-indigo-800">
                <h3 class="text-lg font-semibold">Patient Information & Clinical Metrics</h3>
                <p class="text-xs text-indigo-200 mt-1">Fill out the variables below to evaluate patient outcome classification metrics.</p>
            </div>

            <form action="/predict" method="POST" class="p-8 space-y-6">
                <div class="grid grid-cols-1 md:grid-cols-2 gap-6">
                    
                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-2">Age (Years)</label>
                        <input type="number" name="Age" min="0" max="120" required placeholder="e.g. 45" value="{{ form_values.get('Age', '') }}"
                            class="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all text-slate-800 bg-slate-50/50">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-2">Gender</label>
                        <select name="Gender" required class="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-slate-50/50">
                            <option value="0" {% if form_values.get('Gender') == '0' %}selected{% endif %}>Female</option>
                            <option value="1" {% if form_values.get('Gender') == '1' %}selected{% endif %}>Male</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-2">State (Label Encoded Index)</label>
                        <input type="number" name="State" min="0" required placeholder="e.g. 2" value="{{ form_values.get('State', '') }}"
                            class="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all text-slate-800 bg-slate-50/50">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-2">City (Label Encoded Index)</label>
                        <input type="number" name="City" min="0" required placeholder="e.g. 14" value="{{ form_values.get('City', '') }}"
                            class="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all text-slate-800 bg-slate-50/50">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-2">Cancer Type (Label Encoded Index)</label>
                        <input type="number" name="Cancer_Type" min="0" required placeholder="e.g. 3" value="{{ form_values.get('Cancer_Type', '') }}"
                            class="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all text-slate-800 bg-slate-50/50">
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-2">Cancer Stage</label>
                        <select name="Stage" required class="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-slate-50/50">
                            <option value="0" {% if form_values.get('Stage') == '0' %}selected{% endif %}>Stage I</option>
                            <option value="1" {% if form_values.get('Stage') == '1' %}selected{% endif %}>Stage II</option>
                            <option value="2" {% if form_values.get('Stage') == '2' %}selected{% endif %}>Stage III</option>
                            <option value="3" {% if form_values.get('Stage') == '3' %}selected{% endif %}>Stage IV</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-2">Treatment Type</label>
                        <select name="Treatment_Type" required class="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 bg-slate-50/50">
                            <option value="0" {% if form_values.get('Treatment_Type') == '0' %}selected{% endif %}>Chemotherapy</option>
                            <option value="1" {% if form_values.get('Treatment_Type') == '1' %}selected{% endif %}>Surgery</option>
                            <option value="2" {% if form_values.get('Treatment_Type') == '2' %}selected{% endif %}>Radiation</option>
                        </select>
                    </div>

                    <div>
                        <label class="block text-sm font-medium text-slate-700 mb-2">Survival Timeline (Months)</label>
                        <input type="number" name="Survival_Months" min="0" max="360" required placeholder="e.g. 24" value="{{ form_values.get('Survival_Months', '') }}"
                            class="w-full px-4 py-2.5 rounded-xl border border-slate-200 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500 transition-all text-slate-800 bg-slate-50/50">
                    </div>

                </div>

                <div class="pt-4">
                    <button type="submit" 
                        class="w-full bg-indigo-600 hover:bg-indigo-700 text-white font-semibold py-3 px-6 rounded-xl shadow-md hover:shadow-lg transition-all duration-150 cursor-pointer flex items-center justify-center space-x-2">
                        <span>Execute Analytics Prediction</span>
                        <svg class="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
                    </button>
                </div>
            </form>
        </div>
    </main>

    <footer class="bg-white border-t border-slate-100 py-4 text-center text-xs text-slate-400">
        <p>&copy; 2026 OncoPredict Dashboard System. All rights reserved.</p>
    </footer>

</body>
</html>
"""

# -------------------------------------------------------------------------
# 3. Routing Rules
# -------------------------------------------------------------------------
@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE, form_values={})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Keep track of form values so fields don't clear out after submission
        form_data = request.form.to_dict()

        # Parse numeric and drop-down encoded attributes
        age = float(request.form.get('Age', 0))
        gender = int(request.form.get('Gender', 0))
        state = int(request.form.get('State', 0))
        city = int(request.form.get('City', 0))
        cancer_type = int(request.form.get('Cancer_Type', 0))
        stage = int(request.form.get('Stage', 0))
        treatment_type = int(request.form.get('Treatment_Type', 0))
        survival_months = float(request.form.get('Survival_Months', 0))

        # Shape arguments for the model
        features = np.array([[age, gender, state, city, cancer_type, stage, treatment_type, survival_months]])
        
        # Calculate prognosis status
        prediction = model.predict(features)[0]
        
        # Pull model certainty score if supported
        try:
            probabilities = model.predict_proba(features)[0]
            confidence = round(max(probabilities) * 100, 2)
        except Exception:
            confidence = None

        return render_template_string(
            HTML_TEMPLATE, 
            prediction_text=f"Prediction: {prediction}", 
            confidence=confidence,
            prediction_class=str(prediction).lower(),
            form_values=form_data
        )
        
    except Exception as e:
        return render_template_string(HTML_TEMPLATE, error_text=f"Error processing parameters: {str(e)}", form_values=request.form.to_dict())

if __name__ == '__main__':
    app.run(debug=True)
