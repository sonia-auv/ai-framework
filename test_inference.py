from ultralytics import YOLO
from time import time
import cv2

model = YOLO('yolov8n.pt')

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    t1 = time()
    results = model(frame, imgsz=320, conf=0.5, verbose=False)
    for result in results:

        detection_count = result.boxes.shape[0]

        for i in range(detection_count):
            cls = int(result.boxes.cls[i].item())
            name = result.names[cls]
            confidence = float(result.boxes.conf[i].item())*10
            bounding_box = result.boxes.xyxy[i].cpu().numpy()

            # x = int(bounding_box[0])
            # y = int(bounding_box[1])
            # width = int(bounding_box[2] - x)
            # height = int(bounding_box[3] - y)

            # print('cls',cls)
            # print('name',name)
            # print('confidence',confidence)
            # print('bounding_box',bounding_box)
            # print('')
            cv2.putText(frame, name+' '+"{:.2f}".format(confidence), (int(bounding_box[0])+10,int(bounding_box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2, 1)
            cv2.rectangle(frame, (int(bounding_box[0]),int(bounding_box[1])), (int(bounding_box[2]),int(bounding_box[3])), (0,0,255), 2)
    t2 = time()
    freq = 1/(t2-t1)
    cv2.putText(frame, "{:.2f}".format(freq), (0,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2, 1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
cap.release()
cv2.destroyAllWindows()