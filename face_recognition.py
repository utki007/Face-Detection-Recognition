# Import OpenCV2 for image processing
import cv2

# Import numpy for matrices calculations
import numpy as np

import os
from array import array

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

#array for attendance
a = array("i", [0, 0, 0, 0])

# Create Local Binary Patterns Histograms for face recognization
recognizer = cv2.face.LBPHFaceRecognizer_create()

assure_path_exists("trainer/")

# Load the trained mode
recognizer.read('trainer/trainer.yml')

# Load prebuilt model for Frontal Face
cascadePath = "haarcascade_frontalface_default.xml"

# Create classifier from prebuilt model
faceCascade = cv2.CascadeClassifier(cascadePath);

# Set the font style
font = cv2.FONT_HERSHEY_SIMPLEX

# Initialize and start the video frame capture
cam = cv2.VideoCapture(0)

# Loop
while True:
    # Read the video frame
    ret, im =cam.read()

    # Convert the captured frame into grayscale
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

    # Get all face from the video frame
    faces = faceCascade.detectMultiScale(gray, 1.2,5)

    # For each face in faces
    for(x,y,w,h) in faces:

        # Create rectangle around the face
        cv2.rectangle(im, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 4)

        # Recognize the face belongs to which ID
        Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

        # Check the ID if exist
        if confidence<100:
            if(Id == 35):
                Id = "Utkarsh {0:.2f}%".format(round(100 - confidence, 2))
                a[0]=a[0]+1
            if(Id == 2):
                Id = "Mohit {0:.2f}%".format(round(100 - confidence, 2))
                a[1]=a[1]+1
            if(Id == 39):
                Id = "Rony {0:.2f}%".format(round(100 - confidence, 2))
                a[2]=a[2]+1
            if(Id == 4):
                Id = "MANAN {0:.2f}%".format(round(100 - confidence, 2))
                a[3]=a[3]+1
        else:
            Id = "Unknown {0:.2f}%".format(round(100 - confidence, 2))
        
            
        # Put text describe who is in the picture
        cv2.rectangle(im, (x-22,y-90), (x+w+22, y-22), (0,255,0), -1)
        cv2.putText(im, str(Id), (x,y-40), font, 1, (255,255,255), 3)

    # Display the video frame with the bounded rectangle
    cv2.imshow('im',im) 

    # If 'q' is pressed, close program
    if cv2.waitKey(10) & 0xFF == ord('w'):
        break

# Stop the camera
cam.release()



# Attendance System
students = []

students.append("Utkarsh")
students.append("Mohit")
students.append("Rony")
students.append("Manan")


print("\n")
print("\n")
print("\n")

l=0;


print("##############################")
# Display elements in array.
for value in a:
    if (value>0):
        print(students[l],"Present")
    else:
        print(students[l],"Absent")
    l=l+1
print("##############################")

print("\n")
print("\n")
print("\n")

# Close all windows
cv2.destroyAllWindows()
