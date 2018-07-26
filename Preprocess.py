import os
import numpy as np
from PIL import Image
from skimage import io
from skimage.color import rgb2grey


path=os.getcwd()
#print(path)
address = "/home/joker/Desktop/image_classification/edited/"
f=open('info.txt','r')
cont=f.read()
f.close()
listInfo=cont.split('::::')

X = []
for itr in range(0,int(listInfo[-1])-1):
	filename = address + str(itr) + ".JPG";
	img = io.imread(filename)
	img = rgb2grey(img)
	temp = img.flatten()
#	io.imshow(temp)
	X.append(temp)
X = np.array(X)
np.save('X',X)

#listDir=os.listdir(path+"/edited")
#sorted(listDir , key=numericalSort)
#print(listDir)
'''
X=np.asarray([])
for i in listDir:
    link=path+"/edited"+"/"+i
    img=np.asarray(Image.open(link).convert('LA'))
    #print(img.shape)
    k=img.flatten()
    X=np.append(X,k)
    #print(img.size)
X = np.reshape(X, (-1, 20000))
'''


#print(type(int(listInfo[-1])))

prev=0
#Y = []

Y=np.zeros(int(listInfo[-1])-1)
c=0
for i in range(0,len(listInfo)-1):
	curr=int(listInfo[i])
	#print(curr)
	Y[prev:prev+curr]=c
	curr1=prev+curr
	#print(prev)
	prev=curr1
	#print(prev)
	#print(c)
	c+=1

#np.save('X',X)
np.save('Y',Y)
#for i in Y:
#	print(i)
#print(X)
#print(Y)