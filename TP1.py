import cv2 
import math
import numpy as np
from PIL import Image
import imageio as iio
import sys
import os
import qrcode








def encodeRLE(img,filename='compressed_image.txt'):
    
    encoded = []
    temp = []
    count = 0
    tcount = 0
    prev = None
    size =0
    
    fimg = img.flatten()
    fimg=np.append(fimg,256)
    
    for pixel in fimg:  
            size+=1
            if prev==None :
                prev=pixel
                count+=1
                tcount+=1
            else:
                
                if (count <32767) and (tcount<32767):
                    if count<3:         #cas repetition < 3 
                        if prev!=pixel:
                                
                                temp.append(prev)
                                if size!=len(fimg):
                                    tcount+=1
                                    count=1
                                prev=pixel
                        else:
                                temp.append(prev)
                                if size!=len(fimg):
                                    tcount+=1
                                    count+=1
                                prev=pixel
                    #f"0x{tcount-3:02x}"       
                    else:
                        if count==3:
                            if count!=tcount:                                                              
                                encoded.append(format(tcount-3, '04x'))
                                for i in temp:                                   
                                    encoded.append(format(i, '02x'))
                                temp=[]
                                
                                for i in range(2):
                                    del encoded[len(encoded) -1]
                                
                                tcount=1
    
                        
             
                        temp=[] 
                        if prev!=pixel:                               
                                encoded.append(format(count+32768, '04x'))
                                encoded.append(format(prev, '02x'))
                                count=1
                                tcount=1
                                prev=pixel
                        else:   
                                count+=1
                                tcount=1
                                prev=pixel
                            
                else:
                    if count == 32767:
                        encoded.append(format(count+32768, '04x'))
                        encoded.append(format(prev, '02x'))
                        count=1
                        tcount=1
                        prev=pixel
                    if tcount ==32767:
                        encoded.append(format(tcount, '04x'))
                        for i in temp:
                            encoded.append(format(i, '02x'))
                        temp=[]
                        count=1
                        tcount=1
                        prev=pixel
                
            if size== len(fimg) :
                if tcount > 1 :
                    encoded.append(format(tcount, '04x'))
                    for i in temp:
                                    encoded.append(format(i, '02x'))
                    temp=[]    

                chaine="".join(encoded) 
                with open(filename,"w") as file:
                    file.write(chaine)
                    file.close()            

    return chaine,encoded


def QrCodage (chaine):
    nf=len(chaine)
    ndb=0
    p=0
    qrcodes=[]
    while(ndb<nf):
        p+=400
        if p>nf:
            p=nf
        chaine0=chaine[ndb:p]
        qr0 = qrcode.make(chaine0)
        qrcodes.append(qr0)
        ndb=ndb+400
    return qrcodes


def QrDecodage (liste):
    deco=[]
    for i in liste:
        pil_image = i.convert('RGB') 
        open_cv_image = np.array(pil_image) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image.copy() 
        det = cv2.QRCodeDetector()
        retval, points, straight_qrcode = det.detectAndDecode(open_cv_image)
        deco.append(retval)
    
    decoded=""
    for i in deco:
        decoded+=i
    
    encodedlist=[]
    i=0
    while (i<len(decoded)):
        j=decoded[i:i+4]
        k=int(j,base=16)
        i+=4
        if k<32768:
            encodedlist.append(j)
            p=0
            while p<k:
                encodedlist.append(decoded[i:i+2])
                i+=2
                p+=1  
        else:
            encodedlist.append(j)
            encodedlist.append(encoded[i:i+2])
            i+=2 

    return encodedlist,deco


def decodeChaine(shape,filename='compressed_image.txt'):
    with open(filename, "r") as file:
        encoded = file.readlines()[0]
    encodedlist=[]
    i=0
    while (i<len(encoded)):
        j=encoded[i:i+4]
        k=int(j,base=16)
        i+=4
        if k<32768:
            encodedlist.append(j)
            p=0
            while p<k:
                encodedlist.append(encoded[i:i+2])
                i+=2
                p+=1    
        else:
            encodedlist.append(j)
            encodedlist.append(encoded[i:i+2])
            i+=2
    
    nr=0
    r=0
    t=0
    temp=[]
    size=0
    for step in encodedlist:
        size+=1
        if len(step)>2:        
            if int(step[0],base=16) < 8:
                    nr = int(step, base=16)
            else :
                    r = (int (step,base=16))-32768
             
        else :
                if nr>0:
                    t=int (step,base=16)
                    temp.append(t)
                    nr-=1
                if r>0:
                    while r>0:
                        t=int (step,base=16)
                        temp.append(t)
                        r-=1
    
    if size == len(encodedlist):
        decoded = np.array(temp).reshape(shape).astype(np.uint8)
    
    return decoded
    
    
    
    



#def decodeChaineQR(encoded):
    encodedlist=[]
    i=0
    while (i<len(encoded)):
        j=encoded[i:i+4]
        k=int(j,base=16)
        i+=4
        if k<32768:
            encodedlist.append(j)
            p=0
            while p<k:
                encodedlist.append(encoded[i:i+2])
                i+=2
                p+=1    
        else:
            encodedlist.append(j)
            encodedlist.append(encoded[i:i+2])
            i+=2
    
    return encodedlist


def DecodeList (encoded,shape):
    
   
    
    nr=0
    r=0
    t=0
    temp=[]
    size=0
    for step in encoded:
        size+=1
        if len(step)>2:        
            if int(step[0],base=16) < 8:
                    nr = int(step, base=16)
            else :
                    r = (int (step,base=16))-32768
             
        else :
                if nr>0:
                    t=int (step,base=16)
                    temp.append(t)
                    nr-=1
                if r>0:
                    while r>0:
                        t=int (step,base=16)
                        temp.append(t)
                        r-=1
    
    if size == len(encoded):
        decoded = np.array(temp).reshape(shape).astype(np.uint8)
    
    return decoded




img1= np.array ([[0,0,0,0,0,0],
[0,0,0,0,0,0],
[255,255,255,255,255,255],
[255,255,255,255,255,255],
[0,0,0,0,0,0],
[0,0,0,0,0,0]],dtype=np.uint8)


imggray=cv2.imread('youtube.png',cv2.IMREAD_GRAYSCALE)
_, img2=cv2.threshold(imggray,100,255,cv2.THRESH_BINARY)

             
encoded,enc= encodeRLE(img2,'youtube-compressee.txt')

qrcodes=QrCodage(encoded)

dec,deco=QrDecodage(qrcodes)


# for qr in qrcodes:
#     qr.show()


shape=img2.shape 
decodedimg=DecodeList(dec,shape)
decodedimg2=decodeChaine(shape,'youtube-compressee.txt')


height,width=img2.shape
taille1=height * width
print(taille1)
taille2 = (len(encoded))/2 
print(taille2)

tdc=(1-(taille2/taille1))*100

print("le taux de compression = ",tdc,"%")

error_rate = np.count_nonzero(img2 - decodedimg)
print("taux d'erreurs : ",error_rate)
 
cv2.imshow('originale',img2)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows()
# 
cv2.imshow('decodedQRcode',decodedimg)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows()

cv2.imshow('decodedRLE',decodedimg2)
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows()




