import face_recognition
import cv2
import os,sys
import numpy as np
import os
import argparse
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#Emotion Detection Model
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mode = "display"

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))


def emotion_recog(frame):
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)

    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

    # frame = cv2.imread("image1.jpg")
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    print(faces)
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 255), 3)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        #cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)



    # cv2_imshow(frame)
    return emotion_dict[maxindex]

def read_img(path):
  img=cv2.imread(path)
  (h,w)=img.shape[:2]
  width=500
  ratio=width/float(w)
  height=int(h*ratio)
  return cv2.resize(img,(width,height))

known_encodings=[]
known_names=[]
known_dir='./known/18bmc051'

for file in os.listdir(known_dir):
  img=read_img(known_dir +'/'+file)
  img_enc=face_recognition.face_encodings(img)[0]
  known_encodings.append(img_enc)
  known_names.append(file.split('.')[0])
  
unknown_dir='./unknown'
for file in os.listdir(unknown_dir):
  print("Processing",file)
  img1=read_img(unknown_dir+'/'+file)
  img_enc=face_recognition.face_encodings(img1)[0]
  results=face_recognition.compare_faces(known_encodings,img_enc)
  dist=face_recognition.face_distance(known_encodings, img_enc)
  f=0
  #print(file)

  for i in range(len(results)):
        if results[i]:
            f=1
            name = known_names[i]
            name=name[:len(name)-5]
            min_dist=dist[i]
            min_dist=round((100-min_dist),2)
            min_dist=str(min_dist)
            faces= face_recognition.face_locations(img)[0]
            #cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
            #cv2.putText(img, name, (left+2, bottom+20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            #cv2.putText(img, min_dist, (right-2, bottom+20), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255), 2)
            #cv2.imshow('output',img)
            output = emotion_recog(img)
            print(output,name,min_dist)
            #print(list(faces))
            """cv2.imshow("output",img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()"""
            break;

  if(f==0):
    print("Did not match with the dataset")

  

