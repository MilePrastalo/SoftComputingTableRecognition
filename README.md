# SoftComputingTableRecognition

1) Front-end startup: <br />
  1.1) Navigate Command Promp to Angular application "frontend/tableFrontend". <br />
  1.2) Use command "ng serve" to start the application. <br />
  1.3) Application will be started on port 4200. <br />
  1.4) If app fails to start, due to missing modules, use command "npm install", and rerun the application. <br />
  
2) Back-end startup: <br />
  2.1) To run the application, you must have installed Python 3 with Flask, Flask-Cors and other libraries: Keras, opencv-contrib-python, sklearn, tensorflow, numpy, matlib. Use pip to install these. <br />
  2.2) Default settings are set to use already trained artificial neural network (which is included in this project, "model.h5" and                 "model.json"), if you wish to train it yourself, you must go to "tableRecognition/service/ocr.py" and uncomment the line 316. and
      comment the line 317. <br />
  2.3) To run the application, run file "tableRecognition/app.py" <br />
  
 3) How to use application: <br />
   3.1) After you have started front-end and back-end applications, go to the "http://localhost:4200/" <br />
   3.2) There you will see red button "Select File to Upload". Use it to select the picture you wish to upload. <br />
   3.3) After you have chosen the picture, it should be displayed on the same page, nevermind the rotation of the picture. <br />
   3.4) "Noise removal level" goes from 0 to 9, and it represents how much noise removal should be applied to your picture. <br />
   3.5) Click on the button "Send", to proccess the image. <br />
   3.6) After the image has been successfully proccessed, it should send you to another page, with recreated table from your picture. <br />
   3.7) You can use button "Back" to get back to front page, and choose another picture. <br />
