import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
import math

import sort
from sort import * #sort görüntü işleme kullanılan bir izleme algoritmasıdır


cap = cv2.VideoCapture('traffic_2.mp4')



model = YOLO("yolov8m.pt") #yolo ağırlık dosyasının olduğu PATH'i ekle

classNames = ("person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush")

mask = cv2.imread("mask.png")

#Tracking
tracker = sort.Sort(max_age=20, min_hits=3, iou_threshold=0.3) #içindeki değerler değiştirilebilir
limits = [4,297,673,297] #koordinatları aldık

totalCount = []


while True:
    success, img = cap.read()
    mask = cv2.resize(mask, (img.shape[1], img .shape[0])) #mask ve img i aynı boyutlara getirdik
    imgRegion = cv2.bitwise_and(img,mask) #birlestirdik
    #imgGraphic= cv2.imread("graphic.png", cv2.IMREAD_UNCHANGED) #unchaged ile resimi koru, hiç değiştirme diyoruz
    #img = cvzone.overlayPNG(img, imgGraphic, (0,0)) #görüntüleri üst üste ekledik
    results = model(img, stream=True) #stream ile canlı akış modunda çalışması gerektiğini söyledik

    detections = np.empty((0,5)) #0 satır ve 5 sütun oluşturduk!!


    for r in results:
        boxes = r.boxes #boxes ın koordinatların aldık
        for box in boxes:
            #Sınırlandıran kutu - Bounding Box
            x1, y1, x2, y2 = box.xyxy[0] # koordinatları farklı parametrelere attık
            x1, y1, x2, y2 = int(x1), int(y1),int(x2), int(y2) #değerleri integer a çevirdik
            #cv2.rectangle(img, (x1, y1), (x2,y2), (255,0,255), 3)
            w, h = x2 - x1, y2-y1 #yüksekl,l ve kalınlığı hesapladık


            #Confidence - Güven aralığı
            conf = math.ceil((box.conf[0] * 100)) / 100

            #Class Name - Hangi sınıf dikkate alınacak onu belirtiyoruz
            cls = int(box.cls[0]) #tüm sınıfları dikkate almasını söyledik
            currentClass = classNames[cls] #current şimdiki sınıfı ifade eder

            if currentClass == "car" and conf>0.1: #eğer şimdiki sınıf araba ise ve güven aralığı >0.1 ise devam et
                #cvzone.putTextRect(img, f'{currentClass} {Id}', (max(0, x1), max(35, y1)), scale=0.6, thickness=1, offset=3)  # metinle ilgili bilgiler verdik

                #cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=5)

                currentArray = np.array([x1, y1, x2, y2, conf]) #dizi oluturduk
                detections = np.vstack((detections, currentArray)) #her nesne için dizi oluşturduk


    resultsTracker = tracker.update(detections) #nesneleri sortun update işlevine attık. sürekli listeyi yenileyerek takibi sürdürürüz

    cv2.line(img, (limits[0], limits[1]+120), (limits[2], limits[3]+120),(0,0,255),5) #rasgele bir çizgi çizdik

    for result in resultsTracker: #takip edilen v sürekli yenilenen detections ı result a attık
        x1, y1, x2, y2, id  = result #koordinatlarını ve id lerini aldık
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2) #NaN değerleri verdiği için hata veriyor niye NaN deüerleri veriyor acaba
        print(result)
        w, h = x2-x1, y2-y1 #yeni çerçeveyi belirledik
        cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255)) #dikdörtgen çizdik
        cvzone.putTextRect(img, f'{int(id)}', (max(0, x1), max(35, y1)),
                           scale=2, thickness=3, offset=10)  # metinle ilgili bilgiler verdik

        cx, cy = x1+w//2, y1+h//2 #merkezi belirledik

        cv2.circle(img, (cx,cy), 5, (255,0,255), cv2.FILLED) #merkez noktasına circle çizip içini doldurduk

        if limits[0]< cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15: #her araç geçtiğinde totalCOunt değerini artır
            if totalCount.count(id) ==0: #eğer total count 0 ise gir yani kesinlikle gir
                totalCount.append(id) # her id yi ekle
                cv2.line(img, (limits[0], limits[1]+120), (limits[2], limits[3]+120), (0, 255, 0),
                         5)  # rasgele bir çizgi çizmiştik her araç geçtğinde yeşile çeviriyoruz anlık



    cvzone.putTextRect(img, f' Count: {len(totalCount)}', (50, 50))  # metinle ilgili bilgiler verdik
    #cv2.putText(img, str(len(totalCount)), ((255,100),cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8))

    cv2.imshow("Image", img)
    cv2.imshow("ImageRegion", imgRegion)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()