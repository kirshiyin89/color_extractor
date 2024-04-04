from flask import Flask, render_template, request
import cv2
import numpy as np
import os
import base64

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def get_dominant_colors(image_path, num_colors=5):
    """
    Extracts the dominant colors from an image using K-Means clustering.

    Args:
        image_path: Path to the image file.
        num_colors: Number of dominant colors to extract (default: 5).

    Returns:
        A list of dominant colors as RGB tuples.
    """
    # Read the image
    img = cv2.imread(image_path)

    # Convert the image to RGB and reshape for K-Means
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape((-1, 3))

    # Convert pixels to float32 (ensure data type)
    pixels = np.float32(pixels)

    # Perform K-Means clustering
    _, _, centers = cv2.kmeans(pixels, num_colors, None, criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0), attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS)

    # Get the dominant colors
    dominant_colors = [tuple(color) for color in centers.astype(np.uint8)]

    return dominant_colors

@app.route('/', methods=['GET', 'POST'])
def index():
    preview_image = None
    dominant_colors = None
    num_colors = 5  # Default number of colors

    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')

        num_colors_str = request.form.get('num_colors', '')
        if num_colors_str:
            try:
                num_colors = int(num_colors_str)
            except ValueError:
                return render_template('index.html', error='Invalid number of colors')

        if file:
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            with open(filename, "rb") as img_file:
                preview_image = base64.b64encode(img_file.read()).decode('utf-8')
            dominant_colors = get_dominant_colors(filename, num_colors)
            os.remove(filename)

    return render_template('index.html', preview_image=preview_image, dominant_colors=dominant_colors, num_colors=num_colors)


if __name__ == '__main__':
    app.run(debug=True)

