# This test demonstrates how to use the ALPhotoCapture module.
# Note that you might not have this module depending on your distribution
from naoqi import ALProxy

# Replace this with your robot's IP address
IP = "169.254.215.76"
PORT = 9559

# Create a proxy to ALPhotoCapture
photoCaptureProxy = ALProxy("ALPhotoCapture", IP, PORT)

  
  
photoCaptureProxy.setResolution(2)
photoCaptureProxy.setPictureFormat("jpg")
photoCaptureProxy.takePictures(3, "C:/Users/mcdonaldro/Desktop/tmp", "image")