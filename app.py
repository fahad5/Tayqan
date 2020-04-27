# #!/usr/bin/env python
import numpy as np
import cv2
import face_recognition
from PIL import Image
from PIL import Image, ImageDraw
from flask import Flask
from PIL import Image
from PIL import ImageOps
cam = cv2.VideoCapture(0)
def land_marks(frame):
	# cv2.namedWindow("land dmarks")
	face_landmarks_list = face_recognition.face_landmarks(frame)
	for face_landmarks in face_landmarks_list:
	        # Loop over each facial feature (eye, nose, mouth, lips, etc)
	    for name, list_of_points in face_landmarks.items():

	        hull = np.array(face_landmarks[name])
	        hull_landmark = cv2.convexHull(hull)
	        cv2.drawContours(frame, hull_landmark, -1, (0, 255, 0), 3)

	    cv2.imshow("Frame", frame)





def regtangle(img_name):
	image_of_fahad = face_recognition.load_image_file('/Users/fahadalajmi/Documents/homathon/face_recognition_examples-master/img/known/fahad.JPG')
	fahad_face_encoding = face_recognition.face_encodings(image_of_fahad)[0]

	image_of_mohammed = face_recognition.load_image_file('/Users/fahadalajmi/Documents/homathon/face_recognition_examples-master/img/known/mohammed.jpg')
	mohammed_face_encoding = face_recognition.face_encodings(image_of_mohammed)[0]
		#  Create arrays of encodings and names
	known_face_encodings = [
		  fahad_face_encoding,
		  mohammed_face_encoding		]

	known_face_names = [
		  "Fahad Abdullah Alajmi",
		  "Mohammed Fahad Almubadel"
		]
# Load test image to find faces in
	test_image = face_recognition.load_image_file(img_name)
		# Find faces in test image
	face_locations = face_recognition.face_locations(test_image)
	face_encodings = face_recognition.face_encodings(test_image, face_locations)

		# Convert to PIL format
	pil_image = Image.fromarray(test_image)

		# Create a ImageDraw instance
	draw = ImageDraw.Draw(pil_image)
		# Loop through faces in test image
	for(top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
		matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

		name = "Unknown Person"

		  # If match
		if True in matches:
			first_match_index = matches.index(True)
			name = known_face_names[first_match_index]
		  
		  # Draw box
			draw.rectangle(((left, top), (right, bottom)), outline=(255,255,0))

		  # Draw label
			text_width, text_height = draw.textsize(name)
			draw.rectangle(((left,bottom - text_height - 10), (right, bottom)), fill=(255,255,0), outline=(255,255,0))
			draw.text((left + 6, bottom - text_height - 5), name, fill=(0,0,0))

		del draw

		# Display image
		pil_image.show()

		# Save image
		pil_image.save('identify.jpg')

	



def identify(img_name):
	#image that we have in our database
	old_image = face_recognition.load_image_file('/Users/fahadalajmi/Documents/homathon/face_recognition_examples-master/img/known/fahad.JPG') #user should input his photo
	old_encoding = face_recognition.face_encodings(old_image)[0] #convert old image np array to use it for comparison 


	new_image = face_recognition.load_image_file(img_name) # image from camera 
	new_encoding = face_recognition.face_encodings(new_image)[0]

	# Compare faces
	results = face_recognition.compare_faces(
	    [old_encoding], new_encoding)
	print(results[0]) #if true they are identical
	if results[0]:
	    print('This is Fahad')
	else:
	    print('This is NOT Fahad')







def pull_face(img_name):
	image = face_recognition.load_image_file(img_name)
	face_locations = face_recognition.face_locations(image)
	# face_locations[0][0] = int(face_locations[0][0] + 20)
	print([ x for x in face_locations])
	
	for face_location in face_locations:
	    top, right, bottom, left = face_location 
	    top = int(top - 100 )
	    face_image = image[top:bottom, left:right]
	    pil_image = Image.fromarray(face_image)
	    # pil_image.show()
	    pil_image.save(f'{top}.jpg')
img_counter = 0
while True:
	ret, frame = cam.read()
	# frame = cv2.resize(frame, (0,0), fx=1, fy=1)
	land_marks(frame)
	
	if not ret:
	    break
	k = cv2.waitKey(1)

	if k%256 == 27:
	        # ESC pressed
	    print("Escape hit, closing...")
	    break
	elif k%256 == 32:
	        # SPACE pressed
	        # del face_landmarks_list
	    ret, frame1 = cam.read()
	    global img_name
	    img_name = "opencv_frame_{}.png".format(img_counter)

	    cv2.imwrite(img_name, frame1)
	    print("{} written!".format(img_name))

		

	    img_counter += 1
    
regtangle(img_name)
identify(img_name)
pull_face(img_name)
cam.release()

cv2.destroyAllWindows()
