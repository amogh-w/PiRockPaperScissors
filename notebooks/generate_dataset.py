import cv2

cam = cv2.VideoCapture("http://0.0.0.0:4747/mjpegfeed?640x480")

cv2.namedWindow("test")

img_counter = 0

names = ["rock", "paper", "scissor"]
name_counter = 0

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k % 256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k % 256 == 32:
        # SPACE pressed
        if img_counter >= 100:
            img_counter = 0
            name_counter += 1
        img_name = "{}_{}.png".format(names[name_counter], img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        img_counter += 1


cam.release()

cv2.destroyAllWindows()
