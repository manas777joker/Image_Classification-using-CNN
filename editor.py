import os
from PIL import Image

folder=os.path.join(os.getcwd(),"dataset_image")
listD=os.listdir(folder)
listD.sort()
con=0
timages=0
f=open('info.txt','w')
for i in listD:
	print(i)
	path=os.path.join(folder,i)
	listC=os.listdir(path)
	temp=0
	for j in listC:
		link=os.path.join(path,j)
		img = Image.open(link)
		timages+=1
		temp+=1
		img=img.resize((100,100))
		tempPath=os.path.join(os.getcwd(),"edited")
		mainPath=os.path.join(tempPath,str(con))
		mainPath+=".JPG"
		img.save(mainPath)
		con+=1
	f.write(str(temp))
	f.write('::::')
f.write(str(timages))
f.close()

