# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib
# import os
# from PIL import Image
# from numpy import *
# from sklearn.utils import shuffle


# # print("Output from Python") 

# # # # input image dimensions
# img_rows, img_cols = 200, 200

# # # # number of channels
# img_channels = 1

# path1 = 'E:\\unproc'   

# path2 = 'E:\\tom' 
# listing = os.listdir(path1) 

# # num_samples=len(listing)
# # print(num_samples)








# for file in listing:
#      im = Image.open(path1 + '\\' + file)   
#      img = im.resize((img_rows,img_cols))
#      gray = img.convert('L')           
#      gray.save(path2 +'\\' +  file, "JPEG")

# imlist = os.listdir(path2)
#  #print(imlist[0:9])
# im1 = array(Image.open('E:\\tom' + '\\'+ imlist[208]))
# # # # plt.imshow(im1,cmap="gray")
# # # # plt.show()
# immatrix = array([array(Image.open('E:\\tom'+ '\\' + item)).flatten()
#               for item in imlist],'f') 
# # # print(immatrix.shape)
# label=[]
# for i in range(210):
#     label.append(0)

# for i in range(288):
#     label.append(1)   

# label=np.array(label)
# # len(label)
# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(immatrix,label, test_size=0.3)
# #print("</br>")
# # print(X_train.shape)
# # print("</br>")
# # print(X_test.shape)
# # print("</br>")
# from sklearn.linear_model import LogisticRegression

# # # # for training the model
# classifier= LogisticRegression(random_state=0).fit(X_train, y_train)


# y_pred=[]
# for item in X_test:
#     y_pred.append(classifier.predict(item.reshape(-1,40000)))
#     #print(classifier.predict_proba(item.reshape(-1,40000)))



# accuracy=0
# for i in range(150):
#     if y_pred[i]==y_test[i]:
#         accuracy=accuracy+1
    


# print(accuracy)
# print("</br>")


# path_of='C://Users//DELL//Desktop//images//a3.jpg'

# import cv2

# i=img=cv2.imread(path_of,0)
#  # plt.imshow(i,cmap="gray")
#  # plt.show()


# img=cv2.imread(path_of,0)

# print("</br>")
# print(img.shape)

# image=np.array(cv2.resize(img,(200,200))).reshape(-1,40000)
# # print("</br>")
# # print(image.shape)

# val=classifier.predict_proba(image)
# print("</br>")
# print(val)
# img = img.resize((img_rows,img_cols))
# print("</br>")
# import math
# if val[0][0]>val[0][1]:
#      print("male",val[0][0])
#      print("female",val[0][1])
# else:
#      print("male",val[0][0])
#      print("female",val[0][1])

 
import pickle
filename = 'finalized_model.sav'
#pickle.dump(classifier, open(filename, 'wb'))
 
# some time later...
 
# load the model from disk
loaded_model = pickle.load(open(filename, 'rb'))

path_of_x='C://Users//DELL//Desktop//images//a.jpg'

import cv2
import numpy as np
img=cv2.imread(path_of_x,0)
image=np.array(cv2.resize(img,(200,200))).reshape(-1,40000)
val=loaded_model.predict_proba(image)



print("<center> <h1> Predict the Output </h1> </center>")
print("<hr width=65%> ")
print("<center><h4>Hey Your Guessed gender is on the trainded set is:</h4></center>")

name=""
if val[0][0]>val[0][1]:
    name="Male"
    print("<center> <h4> male Probability is ", val[0][0] ," </h4> </center>")
    print("<center> <h4> female Probability is ",  val[0][1] , " </h4> </center>")
else:
    name="Female"
    print("<center> <h4> male Probability is ", val[0][0] ," </h4> </center>")
    print("<center> <h4> female Probability is ",  val[0][1] , " </h4> </center>")


print("<center> <h4 style='color:red'> Final output is--> "+ name+ " </h4> </center>")





