from flask import Flask, render_template, request, redirect, url_for
import streamlit as st
import os
import pickle
import cv2
import face_recognition
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import db
from datetime import datetime
from EncodeSample import process_images_to_firebase

app = Flask(__name__)

cred = credentials.Certificate("D:/PycharmProjects/IBAMS/Generated_servicekey.json")

if not firebase_admin._apps:
    firebase_admin.initialize_app(cred, {
        'databaseURL': "https://ibams-944ab-default-rtdb.firebaseio.com/",
        'storageBucket': "ibams-944ab.appspot.com"
    })
if os.path.exists("D:/PycharmProjects/IBAMS/Images_Encodings.pkl"):
    file = open("D:/PycharmProjects/IBAMS/Images_Encodings.pkl", "rb")
    encodeImagesKnown = pickle.load(file)
    file.close()
    encodeKnown, studentid = encodeImagesKnown

recognized_students = []

output_dir = "D:/PycharmProjects/IBAMS/Trained_images"

def perform_face_recognition(frame, selected_year, selected_branch):
    # Perform face recognition on the captured image
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(img)

    if not face_locations:
        return  # No faces detected, exit the function

    face_encodings = face_recognition.face_encodings(img, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        # Compare the captured face with known faces
        matches = face_recognition.compare_faces(encodeKnown, face_encoding, tolerance=0.45)
        face_distances = face_recognition.face_distance(encodeKnown, face_encoding)

        if not any(matches):  # No matching faces found
            top, right, bottom, left = face_location
            # Draw a rectangle around the face and display "Unknown"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            continue

        match_index = np.argmin(face_distances)

        if matches[match_index]:
            name = studentid[match_index]
            top, right, bottom, left = face_location

            # Check if the student has already been recognized in this frame
            if name not in recognized_students:
                # Draw a rectangle around the face and display the name
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

                # Update the count of total classes attended by the identified person
                ref = db.reference(f'Students/{selected_year}/{selected_branch}/{name}')
                student_info = ref.get()

                if student_info:
                    student_info['Classes_Attended'] += 1
                    ref.set(student_info)

                recognized_students.append(name)

        else:
            top, right, bottom, left = face_location

            # Draw a rectangle around the face and display "Unknown"
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, "Unknown", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)


def update_total_classes_for_all(selected_year, selected_branch):
    # Update the count of total classes for all students in the database
    students_ref = db.reference(f'Students/{selected_year}/{selected_branch}/')
    students = students_ref.get()

    if students:
        current_datetime = datetime.now().strftime('%d-%m-%Y %H:%M:%S')
        for student_id, student_info in students.items():
            if 'Total_Classes' in student_info:
                student_info['Total_Classes'] += 1
                students_ref.child(student_id).child('Total_Classes').set(student_info['Total_Classes'])
                students_ref.child(student_id).child('lasttime_attendance').set(current_datetime)
                # Add or update the field for the current date and time

            # Calculate attendance percentage for each student
            if 'Classes_Attended' in student_info and 'Total_Classes' in student_info and 'Attendance Percentage' in student_info:
                attendance_percentage = (student_info['Classes_Attended'] / student_info['Total_Classes']) * 100
                students_ref.child(student_id).child('Attendance Percentage').set(attendance_percentage)


def process_uploaded_images(uploaded_files, selected_year, selected_branch):
    recognized_students = []
    for uploaded_file in uploaded_files:
        # Read the uploaded image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        # Perform face recognition on the uploaded image
        perform_face_recognition(image, selected_year, selected_branch)

        # Display the image with face recognition results
        st.image(image, channels="BGR")

        # Save the output image with face recognition results
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_filename = f"{output_dir}/output_{timestamp}.jpg"
        cv2.imwrite(output_filename, image)

    not_recognized_students = set(studentid) - set(recognized_students)
    # mark_manual_attendance(not_recognized_students,selected_year, selected_branch)
    update_total_classes_for_all(selected_year, selected_branch)


def retrieve_student_info(registration_number, year, branch):
    # Retrieve student information based on registration number, year, and branch
    student_ref = db.reference(f'Students/{year}/{branch}')
    students = student_ref.get()

    if students:
        for student_id, student_info in students.items():
            db_registration_number = str(student_info.get('Registration Number')).zfill(8)
            input_registration_number = registration_number.zfill(8)
            if db_registration_number == input_registration_number:
                return student_info

    return None


def capture_and_recognize(selected_year,selected_branch):
    cap = cv2.VideoCapture(0)
    captured_images = 0
    while True:
        # Read the video frame
        ret, frame = cap.read()
        cv2.imshow("Video", frame)

        # Check if 'c' key is pressed to capture an image
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            # Perform face recognition on the captured image
            perform_face_recognition(frame, selected_year,selected_branch)

            # Increment the counter for captured images
            captured_images += 1
            # Generate a unique filename using the current timestamp and the image capture counter
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            output_filename = os.path.join(output_dir, f"output_{timestamp}_{captured_images}.jpg")

            # Save the output image
            cv2.imwrite(output_filename, frame)

            # Display the captured image with face recognition results
            st.image(frame, channels="BGR")

        # Check if 'q' key is pressed to quit capturing images
        elif key & 0xFF == ord('q'):
            print("Exiting capture loop.")
            break

    cap.release()
    cv2.destroyAllWindows()
    not_recognized_students = set(studentid) - set(recognized_students)
    # mark_manual_attendance(not_recognized_students, selected_year, selected_branch)

# def mark_manual_attendance(not_recognized_students,selected_year, selected_branch):
#     with st.form(key='manual_attendance_form'):
#         if not_recognized_students:
#             selected_students = st.multiselect("Select Students", not_recognized_students)
#             if st.form_submit_button(label='Submit'):
#                 st.write("OK")
#                 for selected_student in selected_students:
#                     print(f"Processing {selected_student}")
#                     ref = db.reference(f'Students/{selected_year}/{selected_branch}/{selected_student}')
#                     student_info = ref.get()
#
#                     if student_info:
#                         student_info['Classes_Attended'] += 1
#                         ref.set(student_info)
#                         print(f"Attendance marked for {selected_student}")
#                 st.success("Attendance marked manually for selected students.")
#
#     with st.form(key='no_form'):
#         if st.form_submit_button(label='No'):
#             update_total_classes_for_all(selected_year, selected_branch)

storagebucket = storage.bucket()

@app.route('/')
def student_login():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
    if request.method == 'POST':
        # Access input values from the HTML form
        username = request.form['username']
        year = request.form['studentYear']
        branch = request.form['branch']

        # Retrieve student information based on registration number, year, and branch
        student_info = retrieve_student_info(username, year, branch)

        # Display student information or a message if no student is found
        if student_info:
            return render_template('student_display.html', student_info=student_info)
        else:
            return "No student found with the provided details."

emp_username="employee"
emp_password = "svecw"

@app.route('/employee', methods=['GET', 'POST'])
def employee():
    if request.method == 'POST':
        employee_username = request.form['Employee_username']
        employee_password = request.form['Employee_Password']

        if emp_username==employee_username and emp_password==employee_password:
            return redirect(url_for('upload_images'))
        else:
            error='Invalid credentials'
        return render_template('index.html', error=error)

    return render_template('employee.html')

@app.route('/upload_images',methods=['GET','POST'])
def upload_images():
    if request.method == 'POST':
        selected_year = request.form['year']
        selected_branch = request.form['branch']

        if selected_year and selected_branch:
            uploaded_files = request.files.getlist('images')
            print(uploaded_files)
            process_uploaded_images(uploaded_files, selected_year, selected_branch)
            return render_template('upload_images.html')
    return render_template('employee.html')

admin_username = "admin"
admin_password = "123456"

@app.route('/admin', methods=['GET', 'POST'])
def admin():
    if request.method == 'POST':
        entered_admin_username = request.form['Admin_username']
        entered_admin_password = request.form['Admin_Password']

        if entered_admin_username == admin_username and entered_admin_password == admin_password:
            return redirect(url_for('Choice'))
        else:
            error='Invalid credentials'
        return render_template('index.html', error=error)
    return render_template('admin.html')

@app.route('/Choice', methods=['GET', 'POST'])
def Choice():
    if request.method == 'POST':
        admin_menu = request.form['admin_menu']
        if admin_menu == 'Manage Students':
            return render_template('Manage_students.html')
        elif admin_menu == 'Manage Teachers':
            return render_template('Manage_Teacher.html')
        elif admin_menu == 'Reset Student Data':
            return render_template('Reset_student_data.html')
    return render_template('admin.html')

@app.route('/manage_students', methods=['GET', 'POST'])
def manage_students():
    if request.method == 'POST':
        new_student_name = request.form['new_student_name']
        new_student_registration = request.form['new_student_registration']
        new_student_branch = request.form['new_student_branch']
        new_student_year = request.form['new_student_year']
        new_student_mail = request.form['new_student_mail']

        # Process uploaded images
        uploaded_images = request.files.getlist('student_images')
        print(uploaded_images)
        if uploaded_images:
            process_images_to_firebase(uploaded_images, new_student_name)

            year_folder_path = f"SVEC/{new_student_year}/"
            branch_folder_path = f"SVEC/{new_student_year}/{new_student_branch}/"
            student_folder_path = f"SVEC/{new_student_year}/{new_student_branch}/{new_student_name}/"

            # Create the year folder
            year_folder_blob = storage.bucket().blob(year_folder_path)
            year_folder_blob.upload_from_string("")

            # Create the branch folder
            branch_folder_blob = storage.bucket().blob(branch_folder_path)
            branch_folder_blob.upload_from_string("")

            # Create the student folder
            student_folder_blob = storage.bucket().blob(student_folder_path)
            student_folder_blob.upload_from_string("")

            # Upload the images to the student's folder in Firebase Storage
            i = 1
            for uploaded_image in uploaded_images:
                # Generate a unique image filename
                image_filename = f"{new_student_name}{i}.jpg"
                image_blob = storage.bucket().blob(student_folder_path + image_filename)
                i += 1

                # Save the uploaded image to a local file
                with open(image_filename, 'wb') as f:
                    f.write(uploaded_image.read())

                # Upload the local file to Firebase Storage
                image_blob.upload_from_filename(image_filename)

        # Additional logic to save student data in Firebase
        student_info = {
            "Attendance Percentage": 0,
            "Branch": new_student_branch,
            "Classes_Attended": 0,
            "E-Mail": new_student_mail,
            "Name": new_student_name,
            "Registration Number": new_student_registration,
            "Total_Classes": 0,
            "Year_Study": new_student_year
        }

        ref_students = db.reference(f"Students/{new_student_year}/{new_student_branch}/{new_student_name}")
        ref_students.set(student_info)

        # Redirect or render success page
        return render_template('student.html', message='Student added successfully!')

    return render_template('manage_students.html')

@app.route('/add_teacher', methods=['POST'])
def add_teacher():
    if request.method == 'POST':
        new_teacher_name = request.form['teacher_name']
        new_teacher_id = request.form['teacher_id']
        new_teacher_department = request.form['teacher_department']
        new_teacher_email = request.form['teacher_email']
        new_teacher_mobile = request.form['teacher_mobile']

        # Add logic to store new teacher information in the database
        new_teacher_info = {
            "Name": new_teacher_name,
            "Teacher ID": new_teacher_id,
            "Department": new_teacher_department,
            "Email": new_teacher_email,
            "Mobile": new_teacher_mobile
        }

        ref_teachers = db.reference("Teachers")
        ref_teachers.child(new_teacher_id).set(new_teacher_info)

        return render_template('teacher.html', message='Teacher added successfully!')

@app.route('/reset_student_data', methods=['POST'])
def reset_student_data():
    if request.method == 'POST':
        reset_year = request.form['reset_year']
        reset_branch = request.form['reset_branch']

        # Construct the path to the students in the Firebase Realtime Database
        reset_students_path = f'Students/{reset_year}/{reset_branch}'

        # Reference to all students in the Realtime Database for the specified year and branch
        students_ref = db.reference(reset_students_path)

        # Get all student IDs in the specified year and branch
        student_ids = students_ref.get()

        if student_ids:
            # Loop through all students and reset their data
            for student_id in student_ids:
                student_ref = db.reference(f'{reset_students_path}/{student_id}')
                student_ref.update({
                    'Classes_Attended': 0,
                    'Total_Classes': 0,
                    'Attendance Percentage': 0
                })

            return render_template('reset.html', message=f"Data reset for all students in {reset_year}, {reset_branch}")
        else:
            return render_template('warning.html', message="No students found for the specified year and branch.")


if __name__ == '__main__':
    app.run(debug=True)