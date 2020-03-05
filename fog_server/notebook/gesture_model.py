import glob
import os
import json
import pickle as pkl
import pandas as pd
from scipy.spatial import distance
from sklearn import preprocessing
import numpy as np
from sklearn.metrics import accuracy_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
import joblib

FOLDER_PATH = "/Users/kelvinwang/Downloads/JSON/"
PARENT_DIR = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

static_features = ['nose_x' , 'nose_y' , 'leftEar_x' , 'leftEar_y' , 'rightEar_x' , 'leftEar_y',
                                                                                    'leftEye_x' , 'leftEye_y' ,
                   'rightEye_x' , 'rightEye_y' , 'leftHip_x' , 'leftHip_y' , 'rightHip_x' , 'rightHip_y']
dynamic_features = ['rightElbow_x' , 'rightElbow_y' , 'rightWrist_x' , 'rightWrist_y' , 'rightShoulder_x' ,
                    'rightShoulder_y' , 'leftElbow_x' , 'leftElbow_y' , 'leftWrist_x' , 'leftWrist_y' ,
                    'leftShoulder_x' , 'leftShoulder_y']

mm_scaler = preprocessing.MinMaxScaler()

def read_json_files(path):
    """
    Read the JSON files for coming up with the models
    :param path:
    :return:
    """
    files = glob.glob(path)
    featureRow = []
    for name in files:
        gesture_name = name.replace(FOLDER_PATH , '').split('_')[0].split("/")[0]
        with open(name) as f:

            df = convert_to_flat_records(f.read())
            featureList = process_features(df)
            feature_series = pd.Series(featureList)
            feature_series['gesture'] = gesture_name.lower().strip()
            featureRow.append(feature_series)

    dfObj = pd.DataFrame(featureRow)
    print(dfObj.gesture.unique())
    y = dfObj['gesture']
    X = dfObj.drop('gesture' , axis=1)

    print(X.columns)


    X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.33, random_state=1)

    #print(X_test.value_counts())

    X_train_min_max = mm_scaler.fit_transform(X_train)
    X_test = mm_scaler.transform(X_test)

    # Save the scaler object
    save_to_pickle(PARENT_DIR + "/models/scaler_obj.pkl", mm_scaler)

    # Print the test data gesture value count
    #print(y_test.value_counts())
    create_model("model1" , X_train_min_max , y_train , X_test , y_test)
    create_model("model2" , X_train_min_max , y_train , X_test , y_test)
    create_model("model3" , X_train_min_max , y_train , X_test , y_test)
    create_model("model4" , X_train_min_max , y_train , X_test , y_test)


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
    data = json.loads(data)
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

def create_model(model_name , X_train_min_max , y_train , X_test , y_test):
    """
    Create the models and save them as pickle files
    :param model_name:
    :param X_train_min_max:
    :param y_train:
    :param X_test:
    :param y_test:
    :return:
    """

    if model_name == 'model1':
        # clf = MLPClassifier(solver='adam' , alpha=1e-1 , hidden_layer_sizes=(15 ,) , max_iter=3000 ,
        #                     random_state=3)  # 87.9%
        # params = {'n_estimators': 200 , 'min_samples_split': 10 , 'min_samples_leaf': 2 , 'max_features': 'auto' ,
        #           'max_depth': 20 , 'class_weight': 'balanced_subsample' , 'bootstrap': True}
        #
        # clf = RandomForestClassifier(**params)
        clf = GaussianNB()
    elif model_name == 'model2':
        # clf = MLPClassifier(solver='lbfgs' , alpha=1e-5 , hidden_layer_sizes=(15, 10, 10) , max_iter=5000,
        #                     random_state=4)  # 86.7%
        clf = DecisionTreeClassifier(random_state=42 , max_depth=7)
    elif model_name == 'model3':
        # clf = MLPClassifier(solver='adam' , alpha=1e-6 , hidden_layer_sizes=(15 ,) , max_iter=5000,
        #                     random_state=4)  # 85.5
        clf = svm.SVC(kernel='linear', C=100)
    elif model_name == 'model4':
        clf = MLPClassifier(solver='lbfgs' , alpha=1e-1 , hidden_layer_sizes=(10, 10, 15) , max_iter=5000,
                            random_state=5)  # 85.54%

    clf.fit(X_train_min_max, y_train)

    save_to_pickle(PARENT_DIR + "/models/" + model_name + '.pkl', clf)

    y_pred = clf.predict(X_test)


    accuracy = accuracy_score(y_test , y_pred)

    # for a , p in zip(y_test , y_pred):
    #     print("Actual ={} , Predicted = {}".format(a,p))

    print(classification_report(y_test , y_pred))

    print("Accuracy for {} is {}".format(model_name , accuracy))

    outside_predictions(clf)


    #outside_predictions(clf)

###################################
# Serialize the object using Pickle
###################################
def save_to_pickle(filename , model):
    try:
        f = open(filename , 'wb')  # Pickle file is newly created where current python file is
        pkl.dump(model , f)  # dump data to f
        f.close()
    except Exception as e:
        print("Save to Pickle failed :: "+str(e))
        pass
    finally:
        print("Done with save_to_pickle() ::" + filename)


def process_features(df):
    """
    Perform Feature Extraction using the above static_features & dynamic_features
    Take one static and one dynamic feature and get their euclidean distance
    :param df:
    :return: Feature List for one CSV file
    """
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

    return featureList

def generate_test_json_from_csv():
    path = "/Users/phanimadhavjasthi/Documents/MyWork/ASU_Masters/CSE535_MobileComputing/Project/TestData/test/"
    files = glob.glob(path + "*.csv")
    featureRow = []
    for name in files:
        with open(name) as f:
            file_name = name.replace(path , "").split("_")[0] + ".json"
            df = pd.read_csv(f , sep=",")
            df.to_json(path + file_name , orient='records')




#################REMOVE LATER###############

def convert_data_to_features(testdata):


    #df = pd.read_json(json.dumps(testdata))
    df = convert_to_flat_records(testdata)

    featureList = []
    for dfeat in dynamic_features:
        rDF = df[dfeat]
        for sfeat in static_features:
            rSF1 = df[sfeat]
            dst_rDF_rSF = distance.euclidean(rSF1 , rDF)
            featureList.append(dst_rDF_rSF)
            # This loop is for calculating the dist between static and dynamic point and
            # distance between 2 static point and finally dividing them both
            # for sfeat1 in static_features:
            #     if sfeat != sfeat1:
            #         rSF2 = df[sfeat1]
            #         dst_rDF_rSF = float(distance.euclidean(rSF1 , rDF)) / float(distance.euclidean(rSF1 , rSF2))
            #         featureList.append(dst_rDF_rSF)

    feature_series = pd.Series(featureList)


    #min_max_scalar = joblib.load(CURRENT_PATH + "/models/" + "scaler_obj.pkl")
    X_test = mm_scaler.transform([feature_series])
    output = {}

    for i in range(1,5):
        file_name = PARENT_DIR + "/models/model" + str(i) + ".pkl"
        model = joblib.load(file_name)
        #print(model.classes_)
        #print(model.predict_proba(X_test))
        output[str(i)] = model.predict(X_test)[0]

    return json.dumps(output)

def outside_predictions(clf):

    test_folder  = "/Users/phanimadhavjasthi/Documents/MyWork/ASU_Masters/CSE535_MobileComputing/Project/TestData/test_json/"

    files = glob.glob(test_folder + "*.json")

    for name in files:
        gesture_name = name.replace(test_folder , '').split('_')[0].split("/")[0]
        with open(name) as f:
            x_test_1 = convert_data_to_features(f.read())
            print('Gesture Name is :' + str(gesture_name).lower().strip())
            print(x_test_1)
            print('\n\n\n')


def main():
    read_json_files(FOLDER_PATH + "*/*.json")




if __name__ == '__main__': main()