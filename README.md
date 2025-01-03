# Sort_Memories

## Brief

Sort photos based on the criteria of "Me with my favorite people (x, y, z...)" out of a bunch of group photos/random photos.

## Overview

In scenarios where you have a collection of photos from a public event and want to find yourself or specific individuals:

1. **Upload Sample Pictures** of each individual for face detection on `local_host/upload`.
2. **Upload Group Pictures** for recognition and sorting on `local_host/recog`.

The sorted images will be available in the "saved_data" directory in your current directory.

## Instructions

### 1. Upload Sample Pictures

- Navigate to `local_host/upload`.
- Upload sample images of individuals for face detection.

### 2. Upload Group Pictures

- Navigate to `local_host/recog`.
- Upload group images for face recognition and sorting.

### 3. View Results

- After processing, the sorted images will be available in the `saved_data` directory within your current directory.

## SETUP

### 1. Clone and Navigate to Repository
```bash
git clone https://github.com/Karvy-Singh/Sort_Memories.git
cd Sort_Memories
```
### 2. Set Up Virtual Environment
```bash
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
```
### 3. Install Required Dependencies
```bash
pip install face_recognition dlib flask Werkzeug
```
OR 
```bash
pip install -r requirements.txt
```
### 4. Run the Application
```bash
source env/bin/activate  # On Windows use `env\Scripts\activate`
python final.py
```
visit ```http://localhost:5000``` or any other port as displayed on terminal to run the application and follow the above mentioned steps.


