from flask import Flask, render_template, request, redirect, url_for, session
from werkzeug.utils import secure_filename
import os
import uuid
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications import vgg19
from tensorflow.keras.applications.vgg19 import preprocess_input
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = 'your_secret_key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///site.db'
db = SQLAlchemy(app)

model = vgg19.VGG19(weights='imagenet', include_top=False)
style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(20), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(60), nullable=False)

class StyledImage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    user = db.relationship('User', backref=db.backref('styled_images', lazy=True))

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def style_transfer(content_array, style_array):
    content_feature_maps = model.predict(content_array)
    style_feature_maps = model.predict(style_array)

    style_features = [style_layer[0] for style_layer in style_feature_maps]
    content_features = [content_layer[0] for content_layer in content_feature_maps]

    style_gram_matrices = [tf.linalg.einsum('ijk,ijl->kl', style_feature, style_feature) for style_feature in style_features]
    content_gram_matrices = [tf.linalg.einsum('ijk,ijl->kl', content_feature, content_feature) for content_feature in content_features]

    return style_gram_matrices, content_gram_matrices

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'content' not in request.files or 'style' not in request.files:
        return redirect(request.url)
    content_file = request.files['content']
    style_file = request.files['style']
    if content_file.filename == '' or style_file.filename == '':
        return redirect(request.url)
    if content_file and allowed_file(content_file.filename) and style_file and allowed_file(style_file.filename):
        content_filename = secure_filename(content_file.filename)
        style_filename = secure_filename(style_file.filename)
        content_filepath = os.path.join(app.config['UPLOAD_FOLDER'], content_filename)
        style_filepath = os.path.join(app.config['UPLOAD_FOLDER'], style_filename)
        content_file.save(content_filepath)
        style_file.save(style_filepath)

        content_array = preprocess_image(content_filepath)
        style_array = preprocess_image(style_filepath)

        style_gram_matrices, content_gram_matrices = style_transfer(content_array, style_array)

        styled_image = # Your style transfer code here
        styled_image_filename = 'styled_' + str(uuid.uuid4()) + '.jpg'
        styled_image_path = os.path.join(app.config['UPLOAD_FOLDER'], styled_image_filename)
        styled_image.save(styled_image_path)

        # Save styled image to database
        user_id = session.get('user_id')
        styled_image_entry = StyledImage(filename=styled_image_filename, user_id=user_id)
        db.session.add(styled_image_entry)
        db.session.commit()

        return redirect(url_for('view_styled_image', styled_image_id=styled_image_entry.id))
    else:
        return redirect(request.url)

@app.route('/styled_image/<int:styled_image_id>')
def view_styled_image(styled_image_id):
    styled_image_entry = StyledImage.query.get_or_404(styled_image_id)
    return render_template('styled_image.html', styled_image=styled_image_entry)

if __name__ == '__main__':
    app.run(debug=True)
