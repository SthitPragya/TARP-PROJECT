import cv2
cam = cv2.VideoCapture(0)
harcascadePath = "haarcascade_frontalface_default.xml"
detector=cv2.CascadeClassifier(harcascadePath)
detector.load('haarcascade_frontalface_default.xml')

while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray,1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),4)      
    cv2.imshow('Face recognised',img)
    if cv2.waitKey(100) & 0xFF == ord('q'):
       break
cv2.destroyAllWindows()
