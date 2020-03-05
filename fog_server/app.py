from flask import Flask, jsonify, request
from scipy.spatial import distance
import pandas as pd
import os
import joblib
import json
import numpy as np
import random
from sklearn import preprocessing

app = Flask(__name__)

CURRENT_PATH = os.path.dirname(os.path.realpath(__file__))

output = {}
@app.route('/',methods=['GET','POST'])
def model_predict():
    try:
        if request.method == 'POST':
            test_data = request.get_json(force=True)
            output = convert_data_to_features(test_data)
            resp = app.make_response(output)
            resp.mimetype = "application/json"
            return resp
        else:
            return jsonify("GET is not supported")
    except Exception as e:
        print("Exception while running model_predict()" + str(e))
        raise e



def convert_data_to_features(testdata):


    #df = pd.read_json(json.dumps(testdata))
    df = convert_to_flat_records(testdata)

    static_features = ['nose_x' , 'nose_y' , 'leftEar_x' , 'leftEar_y' , 'rightEar_x' , 'leftEar_y',
                       'leftEye_x', 'leftEye_y','rightEye_x','rightEye_y'
                        , 'leftHip_x' , 'leftHip_y' , 'rightHip_x' , 'rightHip_y']
    dynamic_features = ['rightElbow_x' , 'rightElbow_y' , 'rightWrist_x' , 'rightWrist_y' , 'rightShoulder_x' ,
                        'rightShoulder_y' , 'leftElbow_x' , 'leftElbow_y' , 'leftWrist_x' , 'leftWrist_y' ,
                        'leftShoulder_x' , 'leftShoulder_y']

    featureList = []
    for dfeat in dynamic_features:
        rDF = df[dfeat]
        for sfeat in static_features:
            rSF1 = df[sfeat]
            # This loop is for calculating the dist between static and dynamic point and
            # distance between 2 static point and finally dividing them both
            dst_rDF_rSF = distance.euclidean(rSF1 , rDF)
            featureList.append(dst_rDF_rSF)
            # for sfeat1 in static_features:
            #     if sfeat != sfeat1:
            #         rSF2 = df[sfeat1]
            #         dst_rDF_rSF = float(distance.euclidean(rSF1 , rDF)) / float(distance.euclidean(rSF1 , rSF2))
            #         featureList.append(dst_rDF_rSF)

    feature_series = pd.Series(featureList)

    min_max_scalar = joblib.load(CURRENT_PATH + "/models/scaler_obj.pkl")
    X_test = min_max_scalar.transform([feature_series])

    for i in range(1,5):
        file_name = CURRENT_PATH + "/models/model" + str(i) + ".pkl"
        model = joblib.load(file_name)
        output[str(i)] = model.predict(X_test)[0]

    return json.dumps(output)

def convert_to_flat_records(data):
    columns = ['score_overall' , 'nose_score' , 'nose_x' , 'nose_y' , 'leftEye_score' , 'leftEye_x' , 'leftEye_y' ,
               'rightEye_score' , 'rightEye_x' , 'rightEye_y' , 'leftEar_score' , 'leftEar_x' , 'leftEar_y' ,
               'rightEar_score' , 'rightEar_x' , 'rightEar_y' , 'leftShoulder_score' , 'leftShoulder_x' ,
               'leftShoulder_y' , 'rightShoulder_score' , 'rightShoulder_x' , 'rightShoulder_y' , 'leftElbow_score' ,
               'leftElbow_x' , 'leftElbow_y' , 'rightElbow_score' , 'rightElbow_x' , 'rightElbow_y' ,
               'leftWrist_score' , 'leftWrist_x' , 'leftWrist_y' , 'rightWrist_score' , 'rightWrist_x' ,
               'rightWrist_y' , 'leftHip_score' , 'leftHip_x' , 'leftHip_y' , 'rightHip_score' , 'rightHip_x' ,
               'rightHip_y' , 'leftKnee_score' , 'leftKnee_x' , 'leftKnee_y' , 'rightKnee_score' , 'rightKnee_x' ,
               'rightKnee_y' , 'leftAnkle_score' , 'leftAnkle_x' , 'leftAnkle_y' , 'rightAnkle_score' , 'rightAnkle_x' ,
               'rightAnkle_y']
    #data = json.loads(data)
    csv_data = np.zeros((len(data) , len(columns)))
    for i in range(csv_data.shape[0]):
        one = []
        one.append(data[i]['score'])
        for obj in data[i]['keypoints']:
            one.append(obj['score'])
            one.append(obj['position']['x'])
            one.append(obj['position']['y'])
        csv_data[i] = np.array(one)
    return pd.DataFrame(csv_data, columns=columns)

if __name__ == '__main__':
      app.run(host = "0.0.0.0", port = 5000, debug=True)