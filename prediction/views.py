from django.shortcuts import render
from joblib import load
import pandas as pd
import numpy as np
import json
from django.shortcuts import redirect
from django.core.files.storage import FileSystemStorage
from django.conf import settings
import os
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras_preprocessing.image import load_img, img_to_array
import pytesseract as pt
from .models import ReviewAnalysis,CarDetails,CarModel,CarCompany
from django.http import JsonResponse
import google.generativeai as genai

import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-pro")


car_model = load("./savedModels/RF.joblib")

num_detection_model = tf.keras.models.load_model("./savedModels/object_det_model.h5")
print(num_detection_model, "---------------------------")


with open(
    "./savedModels/prediction_list.json"
) as file:
    data = json.load(file)
    pred_cols = data["pred_list"]



def home(request):
    return render(request, 'home.html')


def predict_price(request):
    if request.method == "POST":
        X_predict = np.zeros(len(pred_cols))

        Company_Name = "Comp_" + request.POST.get("Company_name")
        Model_Name = "Model_" + request.POST.get("Model")
        Engine_Type = request.POST.get("Engine_type")
        Body_Type = request.POST.get("Body_type")
        Transmission_Type = request.POST.get("Transmission_type")
        age_of_car = request.POST.get("Age_of_the_car")
        Kilometers_driven = request.POST.get("Kilometer_driven")

        company_index = pred_cols.index(Company_Name)
        model_index = pred_cols.index(Model_Name)
        age_index = pred_cols.index("age_of_car")
        km_index = pred_cols.index("Kilometers_driven")
        engine_index = pred_cols.index(Engine_Type)
        body_index = pred_cols.index(Body_Type)
        transmission_index = pred_cols.index(Transmission_Type)

        X_predict[company_index] = 1
        X_predict[model_index] = 1
        X_predict[age_index] = age_of_car
        X_predict[km_index] = Kilometers_driven
        X_predict[engine_index] = 1
        X_predict[body_index] = 1
        X_predict[transmission_index] = 1

        predicted_price = int(car_model.predict([X_predict]))

        context = {"predicted_price": predicted_price}

        return render(request, "prediction.html", context)

    return render(request, "prediction.html")


def obj_detection(path):
    # read image
    image = load_img(path)  # object
    image = np.array(image, dtype=np.uint8)  # 8bit array (0,255)
    image1 = load_img(path, target_size=(224, 224))
    image_arr_224 = img_to_array(image1) / 255.0
    h, w, d = image.shape  # rows,colums,and depth

    test_arr = image_arr_224.reshape(1, 224, 224, 3)
    # make predictions
    cordinates = num_detection_model.predict(test_arr)
    # denormalising
    dnorm = np.array([w, w, h, h])
    cordinates = cordinates * dnorm

    cordinates = cordinates.astype(np.int32)

    # draw bounding box on the top of the image
    xmin, xmax, ymin, ymax = cordinates[0]
    pt1 = (xmin, ymin)
    pt2 = (xmax, ymax)

    cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    return image, cordinates


def number_detection(request):
    if request.method == "POST" and request.FILES["file_input"]:
        uploaded_file = request.FILES["file_input"]
        media_root = settings.MEDIA_ROOT
        destination = os.path.join(media_root, "uploaded_images", uploaded_file.name)

        fs = FileSystemStorage(location=media_root)
        fs.save(destination, uploaded_file)
        uploaded_file_url = fs.url(destination)

        img, cods = obj_detection(destination)
        img = np.array(load_img(destination))
        xmin, xmax, ymin, ymax = cods[0]
        roi = img[ymin:ymax, xmin:xmax]
        print(roi, "---roi")
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        # This helps in handling variations in lighting conditions
        thresh_roi = cv2.adaptiveThreshold(
            gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )

        # extract text from image
        text = pt.image_to_string(roi)
        print(text,"----text-->", len(text),'===len')
        
        for i, j in enumerate(text):
            print('index = ',i, str(j), type(str(text)))
            
        text = str(text[:len(text)-2])
        
        
        
        car_details = CarDetails.objects.filter(car_number=text).first()  
        
        print(car_details,'fhfjfjf') 
        
        if not text:
            message = "Sorry, cannot detect number."
            context = {"uploaded_file_url": uploaded_file_url, "number": text, "car_details": car_details, "message": message}
        else:
            context = {"uploaded_file_url": uploaded_file_url, "number": text, "car_details": car_details}

        

        return render(request, "prediction.html", context)
    return render(request, "prediction.html")


def sentimental_analysis(request):
    if request.method == "POST":
        review = request.POST.get("review", "")
        print(review, "----->review")

        prompt = f"""
        You are sentimental analysis model. Analyse the sentence and tell if it is positive or negative:
        \n{review}.
        only return 'This is a Happy customer' if it is positive or 'This is an unhappy customer' if it is negative.
        if review lack context return 'Cannot determine sentiment without more context'.
        """

        print(prompt)
        response = model.generate_content(prompt)

        sentiment = response.text
        print(review, "----->sentiment")

        sentiment_bytes = None

        if sentiment == "This is a Happy customer":
            sentiment_bytes = 1
        else:
            sentiment_bytes = 0

        print(sentiment_bytes, "-------------> sentiment")

        review_obj = ReviewAnalysis(review=review, sentiment=sentiment_bytes)
        review_obj.save()

        return render(
            request, "review.html", {"review": review, "sentiment": sentiment}
        )

    return render(request, "review.html")


def get_model(request):
    selected_company = request.GET.get('company', None)

    # Query your database to get models for the selected company
    models = CarModel.objects.filter(car_company__name=selected_company).values_list('name', flat=True)
    print(models,"ggggggggggggggg")
    return JsonResponse({'models': list(models)})
