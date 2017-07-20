from sklearn.svm import SVC
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn import metrics
import matplotlib.pyplot as plt

def train_and_predict(csv_data, label_column):

    print(csv_data.shape)
    print(csv_data.columns)
    row_number = csv_data.shape[0]
    train_split = csv_data.ix[:int(row_number*0.8)]
    test_split = csv_data.ix[int(row_number*0.8)+1:]

    train_df = train_split.drop(label_column, axis=1)
    print(train_df.shape)
    label = train_split[label_column].values
    model = SVC()
    model.fit(X=train_df, y=label)
    # write model to binary file
    # joblib.dump(model, "SVM_model.pkl")

    test_df = test_split.drop(label_column, axis=1)
    print(test_df.shape)
    answer_label = test_split[label_column].values
    predict_result = model.predict(test_df)
    print(metrics.classification_report(answer_label, predict_result))
    print(metrics.confusion_matrix(answer_label, predict_result))

def main():

    csv_data = pd.read_csv('data/white.csv', sep=";")
    label_column = 'quality'
    for col in csv_data.columns:
        plt.hist(csv_data[col])
        plt.savefig("data/"+col+".jpg")
        # plt.show()

    data_filter = csv_data[label_column] > 4
    csv_data = csv_data[data_filter]
    data_filter = csv_data[label_column] < 8
    csv_data = csv_data[data_filter]

    print("-------------------------------------------Raw Data Predict-------------------------------------")
    train_and_predict(csv_data, label_column)

    feature_data = csv_data
    pearson_matrix = feature_data.corr("pearson")
    columns_list = feature_data.columns.values.tolist()
    drop_feature = []
    print("------------pearson_matrix-------------")
    for colname in columns_list:
        avr_co = pearson_matrix[colname].fillna(0).mean()
        print("%s : %f " % (colname, avr_co))
        if avr_co < 0.15 and colname != label_column:
            drop_feature.append(colname)
            feature_data = feature_data.drop(colname, axis=1)
    print("------------Selected features-------------")
    print(feature_data.columns)

    print("-------------------------------------------Selected Predict-------------------------------------")
    train_and_predict(feature_data, label_column)

if __name__ == "__main__":
    main()