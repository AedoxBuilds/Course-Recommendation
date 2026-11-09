"""
Course recommendation - inference + Flask frontend.

Steps:
  1. Load the saved preprocessor and model
  2. Create new data (from the web form)
  3. Pass the new data through the preprocessor, then to the model
  4. Generate and show the result

Run:   python inference_app.py
Open:  http://127.0.0.1:5000
"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, render_template_string, request
from sklearn import set_config

set_config(transform_output="pandas")   # keep column names, same as training

# ==========================================
# 1. LOAD SAVED PREPROCESSOR + MODEL
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
preprocessor = joblib.load(BASE_DIR / "preprocessor.joblib")
model = joblib.load(BASE_DIR / "course_model.joblib")

# Allowed values (same categories the model was trained on)
OPTIONS = {
    "qualification": ["Matric", "Intermediate", "Diploma", "Bachelors", "Masters"],
    "computer_literacy": ["Beginner", "Intermediate", "Advanced"],
    "has_previous_digital_skill": ["No", "Yes"],
    "previous_skill_area": [
        "Social Media", "Programming", "Photoshop/Canva", "HTML/CSS",
        "Video Editing", "Data/Excel", "Gaming/Modding",
    ],
    "course_duration_months": [1, 2, 3, 4, 6, 8, 10, 12],
    "daily_study_hours": [1, 2, 3, 4, 5, 6],
}
FEATURE_ORDER = list(OPTIONS.keys())


# ==========================================
# 2 + 3 + 4. CREATE NEW DATA -> MODEL -> RESULT
# ==========================================
def predict_course(form):
    """Build one new row from the form, run it through the model, return results."""
    row = {}
    for col in FEATURE_ORDER:
        value = form.get(col, "").strip()
        if col in ("course_duration_months", "daily_study_hours"):
            value = int(value)
        row[col] = value

    # Same rule used in training: no previous skill -> "None"
    if row["has_previous_digital_skill"] == "No":
        row["previous_skill_area"] = "None"

    for col, allowed in OPTIONS.items():
        if col == "previous_skill_area" and row[col] == "None":
            continue
        if row[col] not in allowed:
            raise ValueError(f"Invalid value for {col}: {row[col]}")

    new_data = pd.DataFrame([row], columns=FEATURE_ORDER)
    processed = preprocessor.transform(new_data)
    probabilities = model.predict_proba(processed)[0]

    ranked = sorted(zip(model.classes_, probabilities), key=lambda x: x[1], reverse=True)
    return {
        "course": str(ranked[0][0]),
        "confidence": round(float(ranked[0][1]) * 100, 1),
        "top3": [(str(name), round(float(p) * 100, 1)) for name, p in ranked[:3]],
    }


# ==========================================
# FLASK FRONTEND
# ==========================================
app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Course Recommendation</title>
<style>
  * { box-sizing: border-box; }
  body { margin: 0; font-family: system-ui, Arial, sans-serif; background: #f2f4f8; color: #1f2937; }
  .wrap { max-width: 560px; margin: 40px auto; padding: 0 16px; }
  .card { background: #fff; border-radius: 14px; padding: 28px; box-shadow: 0 4px 18px rgba(0,0,0,.08); }
  h1 { margin: 0 0 4px; font-size: 24px; }
  p.sub { margin: 0 0 22px; color: #6b7280; font-size: 14px; }
  label { display: block; margin: 14px 0 6px; font-size: 14px; font-weight: 600; }
  select { width: 100%; padding: 10px 12px; border: 1px solid #d1d5db; border-radius: 8px; font-size: 15px; background: #fff; }
  select:disabled { background: #f3f4f6; color: #9ca3af; }
  button { width: 100%; margin-top: 24px; padding: 12px; border: 0; border-radius: 8px;
           background: #2563eb; color: #fff; font-size: 16px; font-weight: 600; cursor: pointer; }
  button:hover { background: #1d4ed8; }
  .result { margin-top: 22px; padding: 18px; border-radius: 10px; background: #eff6ff; border: 1px solid #bfdbfe; }
  .result .label { font-size: 13px; color: #1e40af; text-transform: uppercase; letter-spacing: .05em; }
  .result .course { font-size: 26px; font-weight: 700; margin: 4px 0 2px; }
  .bar { margin-top: 10px; font-size: 14px; }
  .bar .track { height: 8px; background: #dbeafe; border-radius: 4px; overflow: hidden; margin-top: 3px; }
  .bar .fill { height: 100%; background: #2563eb; }
  .error { margin-top: 18px; padding: 12px; border-radius: 8px; background: #fee2e2; color: #991b1b; font-size: 14px; }
  .note { margin-top: 14px; font-size: 12px; color: #6b7280; }
</style>
</head>
<body>
<div class="wrap"><div class="card">
  <h1>Course Recommendation</h1>
  <p class="sub">Enter the student's details to get a suggested course.</p>

  <form method="post">
    <label>Qualification</label>
    <select name="qualification">
      {% for o in options.qualification %}<option {{ 'selected' if form.get('qualification') == o }}>{{ o }}</option>{% endfor %}
    </select>

    <label>Computer literacy</label>
    <select name="computer_literacy">
      {% for o in options.computer_literacy %}<option {{ 'selected' if form.get('computer_literacy') == o }}>{{ o }}</option>{% endfor %}
    </select>

    <label>Has previous digital skill?</label>
    <select name="has_previous_digital_skill" id="has_skill">
      {% for o in options.has_previous_digital_skill %}<option {{ 'selected' if form.get('has_previous_digital_skill') == o }}>{{ o }}</option>{% endfor %}
    </select>

    <label>Previous skill area</label>
    <select name="previous_skill_area" id="skill_area">
      {% for o in options.previous_skill_area %}<option {{ 'selected' if form.get('previous_skill_area') == o }}>{{ o }}</option>{% endfor %}
    </select>

    <label>Course duration (months)</label>
    <select name="course_duration_months">
      {% for o in options.course_duration_months %}<option {{ 'selected' if form.get('course_duration_months') == o|string }}>{{ o }}</option>{% endfor %}
    </select>

    <label>Daily study hours</label>
    <select name="daily_study_hours">
      {% for o in options.daily_study_hours %}<option {{ 'selected' if form.get('daily_study_hours') == o|string }}>{{ o }}</option>{% endfor %}
    </select>

    <button type="submit">Recommend a course</button>
  </form>

  {% if error %}<div class="error">{{ error }}</div>{% endif %}

  {% if result %}
  <div class="result">
    <div class="label">Recommended course</div>
    <div class="course">{{ result.course }}</div>
    <div>Model confidence: {{ result.confidence }}%</div>
    {% for name, p in result.top3 %}
      <div class="bar">{{ name }} &mdash; {{ p }}%
        <div class="track"><div class="fill" style="width: {{ p }}%"></div></div>
      </div>
    {% endfor %}
    <div class="note">The model's accuracy on test data is about 41%, so treat this as a suggestion, not a certainty.</div>
  </div>
  {% endif %}
</div></div>

<script>
  const hasSkill = document.getElementById("has_skill");
  const skillArea = document.getElementById("skill_area");
  function sync() { skillArea.disabled = hasSkill.value === "No"; }
  hasSkill.addEventListener("change", sync);
  sync();
</script>
</body>
</html>
"""


@app.route("/", methods=["GET", "POST"])
def index():
    result, error = None, None
    form = request.form if request.method == "POST" else {}
    if request.method == "POST":
        try:
            result = predict_course(form)
        except Exception as exc:
            error = f"Could not generate a result: {exc}"
    return render_template_string(PAGE, options=OPTIONS, form=form, result=result, error=error)


if __name__ == "__main__":
    app.run(debug=False)