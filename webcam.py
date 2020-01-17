
import cv2
size = 4
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
classifiers = cv2.CascadeClassifier('E://gender2//frontalfacewa//haarcascade_frontalface_alt.xml')


while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,0) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (int(im.shape[1] / size), int(im.shape[0] / size)))

    # detect MultiScale / faces 
    faces = classifiers.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        cv2.rectangle(im, (x, y), (x + w, y + h),(0,255,0),thickness=1)
        #Save just the rectangle faces in SubRecFaces
        sub_face = im[y:y+h, x:x+w]
        FaceFileName = './public//' + "satya" + ".jpg"
        cv2.imwrite(FaceFileName, sub_face)

    # Show the image
    cv2.imshow('Satyapal jee',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
        
        
cv2.destroyAllWindows()   
webcam.release()



import pickle
filename = 'finalized_model.sav'
#pickle.dump(classifier, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

path_of_x='E:\gender2\public\satya.jpg'

import cv2
import numpy as np
img=cv2.imread(path_of_x,0)
image=np.array(cv2.resize(img,(200,200))).reshape(-1,40000)
val=loaded_model.predict_proba(image)



name=""
if val[0][0]>val[0][1]:
    name="Male"
   
else:
    name="Female"
    

print("Your Guessed gender is- "+ name)



