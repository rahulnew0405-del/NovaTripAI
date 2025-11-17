# app_main.py  — Groq version with safe fallback
import os
from flask import Flask, render_template, request
# Groq SDK
try:
    from groq import Groq
except Exception:
    Groq = None

app = Flask(__name__, template_folder="templates_main")

# Attempt to create a Groq client if key present and SDK installed
GROQ_KEY = os.environ.get("GROQ_API_KEY")
if Groq and GROQ_KEY:
    client = Groq(api_key=GROQ_KEY)
else:
    client = None

def generate_itinerary_via_groq(prompt_text):
    if not client:
        return (
            "MOCK ITINERARY (no API key)\n\n"
            "Day 1: Arrival, local market visit, beach sunset.\n"
            "Day 2: City tour, museum, food stalls.\n"
            "Tip: Stay hydrated."
        )

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",   # UPDATED MODEL
            messages=[
                {"role": "system", "content": "You are a travel assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=800
        )
        return response.choices[0].message.content

    except Exception as e:
        return f"ERROR: calling Groq API failed: {str(e)}"


@app.route("/", methods=["GET", "POST"])
def home():
    result = None
    if request.method == "POST":
        destination = request.form.get("destination","").strip()
        budget = request.form.get("budget","").strip()
        days = request.form.get("days","").strip()
        trip_type = request.form.get("trip_type","").strip()

        prompt = f"""
Create a concise day-by-day itinerary.
Destination: {destination}
Budget: {budget}
Duration: {days} days
Trip type: {trip_type}

Include:
- Daywise schedule with timings
- 2 food suggestions per day
- Rough cost estimate per day
- One safety tip
Be clear and user-friendly.
"""
        result = generate_itinerary_via_groq(prompt)

    return render_template("index_page.html", result=result)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
