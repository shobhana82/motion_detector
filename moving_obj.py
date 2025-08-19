import cv2

# Open camera
cap = cv2.VideoCapture(0)

# Read the first frame
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Find the difference between two frames
    diff = cv2.absdiff(frame1, frame2)
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours (moving objects)
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 1000:  # ignore small movements
            continue
        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame1, "Motion Detected", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("feed", frame1)

    frame1 = frame2
    ret, frame2 = cap.read()

    if cv2.waitKey(40) == 27:  # press ESC to quit
        break

cap.release()
cv2.destroyAllWindows()
