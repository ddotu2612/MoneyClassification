import numpy as np
import cv2 as cv
import time
import os

# Label: 00000 là ko cầm tiền, còn lại là các mệnh giá
label = "00000"

cap = cv.VideoCapture(0)

# Biến đếm, để chỉ lưu dữ liệu sau khoảng 60 frame, tránh lúc đầu chưa kịp cầm tiền lên
i=0
while(True):
    # Capture frame-by-frame
    #
    i+=1
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv.resize(frame, dsize=None,fx=0.3,fy=0.3)

    # Hiển thị
    cv.imshow('frame',frame)

    # Lưu dữ liệu
    if i>=60 & i <= 1060:
        print("Số ảnh capture = ",i-60)
        # Tạo thư mục nếu chưa có
        if not os.path.exists('data/' + str(label)):
            os.mkdir('data/' + str(label))

        cv.imwrite('data/' + str(label) + "/" + str(i) + ".png",frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()