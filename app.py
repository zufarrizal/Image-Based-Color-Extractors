from flask import Flask, render_template, request, jsonify
from PIL import Image
import io
import numpy as np
from sklearn.cluster import KMeans

app = Flask(__name__)

def extract_colors(image_path, num_colors=5):
    image = Image.open(image_path)
    image = image.resize((150, 150))  # resize image for faster processing
    image_array = np.array(image)
    # Reshape the image array
    reshaped_image_array = image_array.reshape(image_array.shape[0] * image_array.shape[1], 3)
    # Apply KMeans algorithm to extract dominant colors
    kmeans = KMeans(n_clusters=num_colors)
    kmeans.fit(reshaped_image_array)
    # Get the colors
    colors = kmeans.cluster_centers_
    return colors.astype(int)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/extract', methods=['POST'])
def extract():
    file = request.files['file']
    num_colors = int(request.form['num_colors'])
    colors = extract_colors(file)
    return jsonify({'colors': colors.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
