import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import tensorflow as tf
import matplotlib.pyplot as plt
from flask import Flask, render_template, url_for, request
import sqlite3
import cv2
import shutil



app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/userlog', methods=['GET', 'POST'])
def userlog():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']

        query = "SELECT name, password FROM user WHERE name = '"+name+"' AND password= '"+password+"'"
        cursor.execute(query)

        result = cursor.fetchall()

        if len(result) == 0:
            return render_template('index.html', msg='Sorry, Incorrect Credentials Provided,  Try Again')
        else:
            return render_template('userlog.html')

    return render_template('index.html')


@app.route('/userreg', methods=['GET', 'POST'])
def userreg():
    if request.method == 'POST':

        connection = sqlite3.connect('user_data.db')
        cursor = connection.cursor()

        name = request.form['name']
        password = request.form['password']
        mobile = request.form['phone']
        email = request.form['email']
        
        print(name, mobile, email, password)

        command = """CREATE TABLE IF NOT EXISTS user(name TEXT, password TEXT, mobile TEXT, email TEXT)"""
        cursor.execute(command)

        cursor.execute("INSERT INTO user VALUES ('"+name+"', '"+password+"', '"+mobile+"', '"+email+"')")
        connection.commit()

        return render_template('index.html', msg='Successfully Registered')
    
    return render_template('index.html')


@app.route('/userlog.html')
def userlogg():
    return render_template('userlog.html')

@app.route('/developer.html')
def developer():
    return render_template('developer.html')

@app.route('/image', methods=['GET', 'POST'])
def image():
    if request.method == 'POST':
 
        dirPath = "static/images"
        fileList = os.listdir(dirPath)
        for fileName in fileList:
            os.remove(dirPath + "/" + fileName)
        fileName=request.form['filename']
        dst = "static/images"
        

        shutil.copy("Test/"+fileName, dst)
        image = cv2.imread("Test/"+fileName)
        
        #color conversion
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imwrite('static/gray.jpg', gray_image)
        #apply the Canny edge detection
        edges = cv2.Canny(image, 250, 254)
        cv2.imwrite('static/edges.jpg', edges)
        #apply thresholding to segment the image
        retval2,threshold2 = cv2.threshold(gray_image,128,255,cv2.THRESH_BINARY)
        cv2.imwrite('static/threshold.jpg', threshold2)
        # create the sharpening kernel
        kernel_sharpening = np.array([[-1,-1,-1],
                                    [-1, 9,-1],
                                    [-1,-1,-1]])

        # apply the sharpening kernel to the image
        sharpened = cv2.filter2D(image, -1, kernel_sharpening)

        # save the sharpened image
        cv2.imwrite('static/sharpened.jpg', sharpened)


        def segment_tumor(image, lower_gray_threshold, upper_white_threshold, pixel_to_mm_conversion):
            # Read the input image
            #image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
            
            # Thresholding to segment the tumor
            mask = cv2.inRange(image, lower_gray_threshold, upper_white_threshold)
            
            # Find contours in the binary image
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Create a mask for the tumor area
            tumor_mask = np.zeros_like(image)
            cv2.drawContours(tumor_mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
            
            # Overlay the tumor mask on the original image
            tumor_area_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            tumor_area_image[tumor_mask == 255] = [0, 255, 0]  # Mark tumor area in green
            
            # Calculate area of the tumor
            tumor_area_pixel = sum(cv2.contourArea(contour) for contour in contours)
            tumor_area_mm = tumor_area_pixel * pixel_to_mm_conversion ** 2
                 
            # Calculate the diameter of the tumor
            _, radius = cv2.minEnclosingCircle(max(contours, key=cv2.contourArea))
            tumor_diameter_pixel = 2 * radius
            tumor_diameter_mm = tumor_diameter_pixel * pixel_to_mm_conversion
            
            return tumor_area_mm, tumor_diameter_mm, tumor_area_image
        # Example usage
##        image_path = "image.jpg"  # Replace with the actual image path

       
        # Adjust these threshold values based on the characteristics of your image
        lower_gray_threshold = 150
        upper_white_threshold = 200

        # Pixel to millimeter conversion factor
        pixel_to_mm_conversion = 0.1  # Example conversion factor, adjust according to your image

        tumor_area_mm, tumor_diameter_mm, tumor_area_image = segment_tumor(gray_image, lower_gray_threshold, upper_white_threshold, pixel_to_mm_conversion)

        print("Area of Tumor:", tumor_area_mm, "mm")
        print("Diameter of Tumor:", tumor_diameter_mm, "mm")



        
        
        verify_dir = 'static/images'
        IMG_SIZE = 50
        LR = 1e-3
        MODEL_NAME = 'Braintumor-{}-{}.model'.format(LR, '2conv-basic')
    ##    MODEL_NAME='keras_model.h5'
        def process_verify_data():
            verifying_data = []
            for img in os.listdir(verify_dir):
                path = os.path.join(verify_dir, img)
                img_num = img.split('.')[0]
                img = cv2.imread(path, cv2.IMREAD_COLOR)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                verifying_data.append([np.array(img), img_num])
                np.save('verify_data.npy', verifying_data)
            return verifying_data

        verify_data = process_verify_data()
        #verify_data = np.load('verify_data.npy')

        
        tf.compat.v1.reset_default_graph()
        #tf.reset_default_graph()

        convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 3], name='input')

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 128, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 32, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = conv_2d(convnet, 64, 3, activation='relu')
        convnet = max_pool_2d(convnet, 3)

        convnet = fully_connected(convnet, 1024, activation='relu')
        convnet = dropout(convnet, 0.8)

        convnet = fully_connected(convnet, 4, activation='softmax')
        convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')

        model = tflearn.DNN(convnet, tensorboard_dir='log')

        if os.path.exists('{}.meta'.format(MODEL_NAME)):
            model.load(MODEL_NAME)
            print('model loaded!')


        fig = plt.figure()
        
        str_label=" "
        accuracy=""
        for num, data in enumerate(verify_data):

            img_num = data[1]
            img_data = data[0]

            y = fig.add_subplot(3, 4, num + 1)
            orig = img_data
            data = img_data.reshape(IMG_SIZE, IMG_SIZE, 3)
            # model_out = model.predict([data])[0]
            model_out = model.predict([data])[0]
            print(model_out)
            print('model {}'.format(np.argmax(model_out)))

            

            if np.argmax(model_out) == 0:
                str_label = "glioma_tumor "
                print("The predicted image of the glioma_tumor is with a accuracy of {} %".format(model_out[0]*100))
                accuracy="The predicted image of the glioma_tumor is with a accuracy of {}%".format(model_out[0]*100)
                
                
            elif np.argmax(model_out) == 1:
                str_label  = "meningioma_tumor"
                print("The predicted image of the meningioma_tumor is with a accuracy of {} %".format(model_out[1]*100))
                accuracy="The predicted image of the meningioma_tumor is with a accuracy of {}%".format(model_out[1]*100)
                
                
            elif np.argmax(model_out) == 2:
                str_label = "pituitary_tumor"
                print("The predicted image of the pituitary_tumor is with a accuracy of {} %".format(model_out[2]*100))
                accuracy="The predicted image of the pituitary_tumor is with a accuracy of {}%".format(model_out[2]*100)
                            
            

            elif np.argmax(model_out) == 3:
                str_label = "Normal"
                print("accuracy of {} %".format(model_out[3]*100))
                accuracy="The predicted image of the Normal is with a accuracy of {}%".format(model_out[3]*100)

            area_count = ["{:.2f}".format(tumor_area_mm), "{:.2f}".format(tumor_diameter_mm),tumor_area_image]
            #Result
            print(f"Area of Tumor: {tumor_area_mm:.2f}","mm")
            print(f"Diameter of Tumor: {tumor_diameter_mm:.2f}","mm")
            cv2.imwrite("static/tumor_area.jpg", tumor_area_image)
            



            


        return render_template('results.html', status=str_label,accuracy=accuracy,area_count=area_count,ImageDisplay="http://127.0.0.1:5000/static/images/"+fileName,ImageDisplay1="http://127.0.0.1:5000/static/gray.jpg",ImageDisplay2="http://127.0.0.1:5000/static/edges.jpg",ImageDisplay3="http://127.0.0.1:5000/static/threshold.jpg",ImageDisplay4="http://127.0.0.1:5000/static/sharpened.jpg",ImageDisplay5="http://127.0.0.1:5000/static/tumor_area.jpg")
        
    return render_template('index.html')

@app.route('/logout')
def logout():
    return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
