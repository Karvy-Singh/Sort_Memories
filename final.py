from flask import Flask, jsonify, request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
import os
import shutil
import numpy as np
import face_recognition

facerecog = Flask(__name__)
CORS(facerecog)

UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
UPLOAD_FOLDER_2 = os.path.join(os.getcwd(), 'uploads2')
SAVED_DATA_FOLDER = os.path.join(os.getcwd(), 'saved_data')
KNOWN_ENCODINGS_PATH = 'known_encodings.npy'
KNOWN_LABELS_PATH = 'known_labels.npy'

# Create folders if they don't exist
for folder in [UPLOAD_FOLDER, UPLOAD_FOLDER_2, SAVED_DATA_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)

@facerecog.route('/upload', methods=['GET','POST'])
def upload_file():
    files = request.files.getlist('files[]')
    if len(files) > 0:
        features = []
        labels = []
        label_count = 0
        print("encoding faces from 'uploads' folder")
        for f in files:
            filename = secure_filename(f.filename)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if not os.path.exists(file_path):
                f.save(file_path)
                image = face_recognition.load_image_file(file_path)
                face_locations = face_recognition.face_locations(image)
                if len(face_locations) > 0:
                    # Pick largest face by area
                    face_locations = sorted(face_locations, key=lambda loc: (loc[2]-loc[0])*(loc[1]-loc[3]), reverse=True)
                    top, right, bottom, left = face_locations[0]
                    face_encoding = face_recognition.face_encodings(image, [face_locations[0]])[0]
                    features.append(face_encoding)
                    label_count += 1
                    labels.append(label_count)

        if len(features) > 0:
            np.save(KNOWN_ENCODINGS_PATH, features)
            np.save(KNOWN_LABELS_PATH, labels)
            print(f"Saved {len(features)} face encodings.")
        else:
            print("No faces found in the 'uploads' folder.")

        shutil.rmtree(UPLOAD_FOLDER)
        return redirect(url_for("facerecog_process"))
    return render_template('face_detect.html')

@facerecog.route('/recog', methods=['GET','POST'])
def facerecog_process():
    if not os.path.exists(KNOWN_ENCODINGS_PATH) or not os.path.exists(KNOWN_LABELS_PATH):
        return render_template('face_recog.html')

    known_encodings = np.load(KNOWN_ENCODINGS_PATH, allow_pickle=True)
    known_labels = np.load(KNOWN_LABELS_PATH, allow_pickle=True)
    print(known_labels)
    files = request.files.getlist('files[]')
    #l=[]
    for f in files:
        filename = secure_filename(f.filename)
        file_path = os.path.join(UPLOAD_FOLDER_2, filename)
        f.save(file_path)
        if os.path.isfile(file_path):
            try:
                image = face_recognition.load_image_file(file_path)
                face_locations = face_recognition.face_locations(image)

                if len(face_locations) > 0:
                    face_enc = face_recognition.face_encodings(image, face_locations)
                   # matched_labels = set()  
                    l=[]
                    c = 0  #counter for matched faces

                for face_encoding, (top, right, bottom, left) in zip(face_enc, face_locations):
                    # Compare against known encodings
                    matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
                    distances = face_recognition.face_distance(known_encodings, face_encoding)

                    if True in matches:
                            best_match_index = np.argmin(distances)
                            # Check distance threshold for the best match
                            if distances[best_match_index] < 0.5:
                                #matched_labels.add(known_labels[best_match_index])
                                c += 1
                                l.append(c)
                                print(f"Match found for {filename}: Label {known_labels[best_match_index]}")

                # Save the image only if both known faces are detected
                #required_labels = {1, 2} 
                if set(l) == set(known_labels):  
                    shutil.copy(file_path, os.path.join(SAVED_DATA_FOLDER, filename))
                    print(f"Image saved: {filename}")

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")

    return render_template('face_recog.html')

if __name__ == '__main__':
    facerecog.run(debug=True)

