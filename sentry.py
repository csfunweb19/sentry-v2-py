

auth_info = wyzecam.login(os.environ["WYZE_EMAIL"], os.environ["WYZE_PASSWORD"])
account = wyzecam.get_user_info(auth_info)
camera = wyzecam.get_camera_list(auth_info)[0]

with wyzecam.WyzeIOTC() as wyze_iotc:
  with wyze_iotc.connect_and_auth(account, camera) as sess:
    for (frame, frame_info) in sess.recv_video_frame_ndarray():
      cv2.imshow("Video Feed", frame)
      cv2.waitKey(1)


import os
import cv2
import wyzecam
import turtle
import tempfile

# # Set up the Turtle screen
screen = turtle.Screen()
screen.title("WyzeCam Video on Turtle Screen")
screen.bgcolor("black")  # Set background color to black for better contrast

# Initialize OpenCV video capture (replace with your actual video file)
video_path = 'wyzecam_video.mp4'  # Replace with your video path
cap = cv2.VideoCapture(video_path)

# Ensure Turtle updates are manual
screen.tracer(0)  # Turn off automatic screen updates
frame_rate = int(cap.get(cv2.CAP_PROP_FPS))

# Define a function to display the video in Turtle
def display_video():
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Error: Couldn't read frame.")
            break

        # Convert the frame to RGB format (Turtle works with RGB)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Save frame as a temporary .gif file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.gif') as temp_file:
            temp_filename = temp_file.name
            cv2.imwrite(temp_filename, frame)  # Save the frame as a .gif
            
            # Set the background image to the frame
            screen.bgpic(temp_filename)

            # Update the Turtle screen with the new frame
            screen.update()

            # Pause to control the frame rate
            turtle.delay(int(1000 / frame_rate))  # Adjust the delay based on frame rate

            # Clean up by deleting the temporary file
            os.remove(temp_filename)

# Call the function to display the video
display_video()

# Release the video capture when done
cap.release()

# Keep the window open
screen.mainloop()

import cv2
print(cv2.__verson__)