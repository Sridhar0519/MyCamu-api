import cv2
import os,sys
import numpy as np
import os


name = input("Enter Roll No: ")
directory="known/"+name
  
# Parent Directory path
parent_dir = "C:/Users/dhars/Desktop/Face_recg"
  
# Path
path = os.path.join(parent_dir, directory)

try:  
    os.mkdir(path)
except:
    pass

print("Directory '%s' created" %directory)
os.chdir(path)


cwd = os.getcwd()
  
# print the current directory
print("Current working directory is:", cwd)
vid_path="C:/Users/dhars/Desktop/Face_recg/Videos/"+name+".mp4"

cap = cv2.VideoCapture(vid_path)
count = 0;
os.chdir=("/content/sri_k")
while count<4:
    count = count + 1
    print(count)
    ret, frame = cap.read()
    
    if (frame is None) == False:
        (h,w)=frame.shape[:2]
        width=500
        ratio=width/float(w)
        height=int(h*ratio)
        frame=cv2.resize(frame,(width,height))
        if count < 10:
            cv2.imwrite(name+"0000" + str(count) + '.jpg', frame)
        elif count < 100:
            cv2.imwrite(name+"000" + str(count) + '.jpg', frame)
        elif count < 1000:
            cv2.imwrite(name+"00" + str(count) + '.jpg', frame)
        elif count < 10000:
            cv2.imwrite(name+"0" + str(count) + '.jpg', frame)
    else:
        break
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
cap.release()
# cv2.destroyAllWindows()