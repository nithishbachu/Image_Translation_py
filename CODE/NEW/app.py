from flask import Flask, render_template, request, jsonify, flash, redirect, url_for
import os
import cv2

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'supersecretkey'

# Global users dictionary for storing registered users (in-memory storage)
users = {}

# Image Processing Function
def process_image(image_path, operation):
    img = cv2.imread(image_path)

    if img is None:
        raise ValueError("Could not read the uploaded image.")

    if operation == 'rgb':
        processed_img = img
    elif operation == 'hsv':
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    elif operation == 'ycbcr':
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    elif operation == 'hls':
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    elif operation == 'xyz':
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    elif operation == 'canny':
        # Convert image to grayscale and apply Gaussian blur before Canny edge detection
        blurred_img = cv2.GaussianBlur(img, (5, 5), 0)
        processed_img = cv2.Canny(blurred_img, 100, 200)
    else:
        raise ValueError(f"Unsupported operation: {operation}")

    # Save the processed image
    base, ext = os.path.splitext(image_path)
    processed_image_path = f"{base}_{operation}{ext}"
    cv2.imwrite(processed_image_path, processed_img)

    return processed_image_path

# Routes
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle POST logic here
        flash("POST request received!", "success")
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users and users[username] == password:
            flash("Login successful!", "success")
            return redirect(url_for('index'))  # Redirect to the index route
        else:
            flash("Invalid username or password!", "danger")
            return redirect(url_for('login'))  # Redirect back to login on failure

    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if username in users:
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))
        
        users[username] = password
        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))
    
    return render_template('register.html')

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400

    file = request.files['image']
    operation = request.form.get('operation', 'hsv')  # Default to HSV if no operation is provided

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    try:
        # Process the image
        processed_image_path = process_image(filepath, operation)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'processed_image': processed_image_path})

@app.route('/chart')
def chart():
    return render_template('chart.html')

@app.route('/performance')
def performance():
    return render_template('performance.html')

if __name__ == '__main__':
    app.run(debug=True)
