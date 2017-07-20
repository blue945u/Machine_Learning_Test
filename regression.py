import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from pylab import polyfit, poly1d
from sklearn import linear_model

def train_and_predict(csv_data, label_column):

    print(csv_data.shape)
    print(csv_data.columns)
    row_number = csv_data.shape[0]
    train_split = csv_data.ix[:int(row_number*0.8)]
    test_split = csv_data.ix[int(row_number*0.8)+1:]

    train_df = train_split.drop(label_column, axis=1)
    print(train_df.shape)
    label = train_split[label_column].values

    # Create linear regression object
    model = linear_model.LinearRegression()
    # Train the model using the training sets
    model.fit(X=train_df, y=label)

    # write model to binary file
    # joblib.dump(model, "regression_model.pkl")

    test_df = test_split.drop(label_column, axis=1)
    print(test_df.shape)
    answer_label = test_split[label_column].values

    # The coefficients
    print('Coefficients: \n', model.coef_)
    predicted_result = model.predict(X=test_df)
    # The mean squared error
    print("predict   answer   difference")
    for index, predict in enumerate(predicted_result[:10]):
        print("%.2f  %.2f  %.2f" % (predict, answer_label[index], predict - answer_label[index]))
    print("Mean difference: %.2f"
          % np.mean((abs(predicted_result - answer_label))))
    # Explained variance score: 1 is perfect prediction
    # print('Variance score: %.2f' % model.score(test_df, answer_label))
    # Plot outputs
    for column in test_df.columns:
        fig, ax = plt.subplots()
        fit = np.polyfit(test_df[column], predicted_result, deg=1)
        ax.plot(test_df[column], fit[0] * test_df[column] + fit[1], color='red')
        ax.scatter(test_df[column], answer_label)
        fig.show()
        fig.savefig("data/pic/regression_" + column + ".jpg")

def main():

    csv_data = pd.read_csv('data/white.csv', sep=";")
    label_column = 'quality'

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