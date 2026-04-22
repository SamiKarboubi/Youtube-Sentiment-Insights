from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import pickle
import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
from dotenv import load_dotenv
import os
import requests

load_dotenv()

app = Flask(__name__)
CORS(app)



def preprocess_comment(comment):

    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment
    

def load_model_and_vectorizer(model_name,model_version,vectorizer_path):

    mlflow.set_tracking_uri('http://ec2-15-188-10-108.eu-west-3.compute.amazonaws.com:5000/')
    
    model_uri = f"models:/{model_name}/{model_version}"

    model = mlflow.pyfunc.load_model(model_uri)

    with open(vectorizer_path,'rb') as file:
        vectorizer = pickle.load(file)

    return model,vectorizer


model, vectorizer = load_model_and_vectorizer('youtube_comments_model','1','../tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return "Welcome to our flask api"

@app.route('/search',methods=['POST'])
def search():
    api_key = os.getenv("YOUTUBE_API_KEY")
    data = request.get_json()
    max_comments = data.get('MAX_COMMENTS')
    next_page_token = data.get('nextPageToken')
    video_id = data.get('videoId')
    url = f"https://www.googleapis.com/youtube/v3/commentThreads?part=snippet&videoId={video_id}&maxResults={max_comments}&key={api_key}"
    if next_page_token:
        url += f"&pageToken={next_page_token}"
    response = requests.get(url)
    if response.status_code != 200:
        return jsonify({"error": "YouTube API error"}), response.status_code
    
    data = response.json()
    return jsonify(data)

@app.route('/video-info', methods=['POST'])
def video_info():
    api_key = os.getenv("YOUTUBE_API_KEY")
    data = request.get_json()
    video_id = data.get('videoId')
    
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet,statistics&id={video_id}&key={api_key}"
    response = requests.get(url)
    
    if response.status_code != 200:
        return jsonify({"error": "YouTube API error"}), response.status_code
    
    data = response.json()
    if not data.get('items'):
        return jsonify({"error": "Video not found"}), 404
    
    return jsonify(data.get('items')[0])

@app.route('/predict',methods=['POST'])
def predict():
    data = request.get_json()
    comments = data.get('comments')

    if not comments:
        return jsonify({"error":"No comments provided"}), 400
    
    try:
        
        preprocessed_comments = [preprocess_comment(comment) for comment in comments]
        transformed_comments = vectorizer.transform(preprocessed_comments)
        dense_comments = transformed_comments.toarray()
        feature_names = vectorizer.get_feature_names_out()
        df_comments = pd.DataFrame(dense_comments, columns=feature_names)
        
        predictions = model.predict(df_comments)
        predictions = [int(pred) if isinstance(pred, (int, np.integer)) else float(pred) for pred in predictions]
    except Exception as e:
        return jsonify({"error" : f"Prediction failed : {str(e)}"}) , 500

    response = [{"comment":comment,"sentiment":sentiment} for comment, sentiment in zip(comments,predictions)]
    return jsonify(response)


@app.route('/generate_chart', methods=['POST'])
def generate_chart():
    try:
        data = request.get_json()
        sentiment_counts = data.get('sentiment_counts')
        
        if not sentiment_counts:
            return jsonify({"error": "No sentiment counts provided"}), 400

        # Prepare data for the pie chart
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")
        
        colors = ['#36A2EB', '#C9CBCF', '#FF6384']  # Blue, Gray, Red

        # Generate the pie chart
        plt.figure(figsize=(6, 6))
        plt.pie(
            sizes,
            labels=labels,
            colors=colors,
            autopct='%1.1f%%',
            startangle=140,
            textprops={'color': 'w'}
        )
        plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

        # Save the chart to a BytesIO object
        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        # Return the image as a response
        return send_file(img_io, mimetype='image/png')
    
    except Exception as e:
        app.logger.error(f"Error in /generate_chart: {e}")
        return jsonify({"error": f"Chart generation failed: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0',port=5000,debug=True)