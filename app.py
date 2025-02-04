from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the scaler and model separately
scaler = pickle.load(open(r"C:\Users\Komal\Desktop\career_rec\model\scaler.pkl", 'rb'))
model = pickle.load(open(r"C:\Users\Komal\Desktop\career_rec\model\model.pkl", 'rb'))
class_names = ['Lawyer', 'Doctor', 'Government Officer', 'Artist', 'Unknown',
               'Software Engineer', 'Teacher', 'Business Owner', 'Scientist',
               'Banker', 'Writer', 'Accountant', 'Designer',
               'Construction Engineer', 'Game Developer', 'Stock Investor',
               'Real Estate Developer']


# Recommendations ===========================================================
def Recommendations(gender, part_time_job, absence_days, extracurricular_activities,
                    weekly_self_study_hours, math_score, history_score, physics_score,
                    chemistry_score, biology_score, english_score, geography_score,
                    total_score, average_score):
    # Encode categorical variables
    gender_encoded = 1 if gender.lower() == 'female' else 0
    part_time_job_encoded = 1 if part_time_job else 0
    extracurricular_activities_encoded = 1 if extracurricular_activities else 0


    feature_array = np.array([[
        gender_encoded, part_time_job_encoded, absence_days, extracurricular_activities_encoded,
        weekly_self_study_hours, math_score,history_score, physics_score,
        chemistry_score, biology_score, english_score,geography_score,
        total_score, average_score
    ]])

    # Scale the features using the same scaler that was used to train the model
    scaled_features = scaler.transform(feature_array)

    # Predict using the model
    probabilities = model.predict_proba(scaled_features)

    # Get top predicted classes along with their probabilities
    top_classes_idx = np.argsort(-probabilities[0])[:3]
    top_classes_names_probs = [(class_names[idx], probabilities[0][idx]) for idx in top_classes_idx]

    return top_classes_names_probs


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/recommend')
def recommend():
    return render_template('recommend.html')


@app.route('/pred', methods=['POST', 'GET'])
def pred():
    if request.method == 'POST':
        # Retrieve and convert form data correctly
        gender = request.form['gender']
        part_time_job = request.form.get('part_time_job') == 'true'  # Ensure the correct conversion to boolean
        absence_days = int(request.form['absence_days'])
        extracurricular_activities = request.form.get(
            'extracurricular_activities') == 'true'  # Ensure the correct conversion
        weekly_self_study_hours = int(request.form['weekly_self_study_hours'])
        math_score = int(request.form['math_score'])
        history_score = int(request.form['history_score'])
        physics_score = int(request.form['physics_score'])
        chemistry_score = int(request.form['chemistry_score'])
        biology_score = int(request.form['biology_score'])
        english_score = int(request.form['english_score'])
        geography_score = int(request.form['geography_score'])
        total_score = float(request.form['total_score'])
        average_score = float(request.form['average_score'])

        # Get recommendations
        recommendations = Recommendations(
            gender, part_time_job, absence_days, extracurricular_activities,
            weekly_self_study_hours, math_score, history_score, physics_score,
            chemistry_score, biology_score, english_score, geography_score,
            total_score, average_score
        )

        return render_template('results.html', recommendations=recommendations)

    return render_template('home.html')


if __name__ == '__main__':
    app.run(debug=True)
