import pandas as pd
import numpy as np
from sklearn import svm
from sklearn import metrics
import matplotlib.pyplot as plt
from pylab import polyfit, poly1d

def train_and_predict(csv_data, label_column):

    data_filter = csv_data[label_column] > 4
    normal_data = csv_data[data_filter]
    data_filter = normal_data[label_column] < 8
    normal_data = normal_data[data_filter]
    normal_data.loc[:, label_column] = 1

    data_filter = csv_data[label_column] <= 4
    abnormal_data = csv_data[data_filter]
    data_filter = csv_data[label_column] >= 8
    abnormal_data = abnormal_data.append(csv_data[data_filter])
    abnormal_data.loc[:, label_column] = -1

    test_split = abnormal_data
    print(test_split.shape)
    normal_data_split = normal_data.ix[:abnormal_data.shape[0]]
    print(normal_data_split.shape)
    test_split = test_split.append(normal_data_split)

    train_split = normal_data.ix[abnormal_data.shape[0]+1:]

    train_df = train_split.drop(label_column, axis=1)
    print(train_df.shape)
    model = svm.OneClassSVM()
    model.fit(X=train_df)

    test_df = test_split.drop(label_column, axis=1)

    answer_label = test_split[label_column].values.astype(int)
    predict_result = model.predict(test_df)

    print(metrics.classification_report(answer_label, predict_result))
    print(metrics.confusion_matrix(answer_label, predict_result))

def main():

    csv_data = pd.read_csv('data/white.csv', sep=";")
    label_column = 'quality'

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