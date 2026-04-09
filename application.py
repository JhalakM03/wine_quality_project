from flask import Flask, request, render_template
import pandas as pd

from src.pipelines.prediction_pipeline import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict_form", method = ["GET", "POST"])

def predict_form():

    if request.method =="GET":
        return render_template("form.html")
    
    try:
        data = request.form

        df = pd.DataFrame([data])

        df.columns = df.colums.str.replace("_"," ")
        df = df.astype(float)

        pipeline = PredictionPipeline()
        prediction = pipeline.predict(df)

        result = "Good Quality" if int(prediction[0]) == 1 else "Bad Quality"

        return render_template("form.html", result=result)
    
    except Exception as e:
        return f"Error: {str(e)}"
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
