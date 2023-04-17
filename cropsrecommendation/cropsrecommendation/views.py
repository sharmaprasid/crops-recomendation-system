
from sklearn.metrics import accuracy_score
from django.shortcuts import render
from public.algorithm.svm import SVM
from public.algorithm.rf import RandomForest


import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def home(request):
    return render(request, 'home.html')


def about(request):
    return render(request, 'about.html')


def predict(request):

    data = pd.read_csv(
        r"E:\cropsdjango\cropsrecommendation\cropsrecommendation\Crop_recommendation.csv")
    X = data.iloc[:, :-1].values
    y = data.iloc[:, -1].values
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=123)

    svm_model = SVM()
    svm_model.fit(X_train, y_train)
    rf_model = RandomForest()

    rf_model.fit(X_train, y_train)
    N = float(request.POST['N'])
    P = float(request.POST['p'])
    K = float(request.POST['k'])
    Temperature = float(request.POST['temperature'])
    Humidity = float(request.POST['humidity'])
    ph = float(request.POST['ph'])
    Rainfall = float(request.POST['rainfall'])
    svm_pred1 = svm_model.predict(X_test)
    rf_pred1 = rf_model.predict(X_test)
    pred1 = svm_model.predict(
        [[N, P, K, Temperature, Humidity, ph, Rainfall]])

    pred2 = rf_model.predict(
        [[N, P, K, Temperature, Humidity, ph, Rainfall]])
    svm_pred = le.inverse_transform(pred1)[0]
    rf_pred = le.inverse_transform(pred2)[0]
    svm_acc = accuracy_score(y_test, svm_pred1)

    rf_acc = accuracy_score(y_test, rf_pred1)

    return render(request, 'home.html', {"predict_rf": rf_pred, "predict_svm": svm_pred, "accuracy_svm": svm_acc, "accuracy_rf": rf_acc})
