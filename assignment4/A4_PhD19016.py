
import pandas as pd
import operator
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(123)
def linear_regressor_closed_form(X,Y):
    return (np.linalg.pinv(np.matmul(X.T,X))@X.T@Y)

#credits I used multiple loops first,  changed to recursion inspired by a thread online
def polynomial_features_with_Interaction(input_rows, degree):
    #base case
    if degree==1:
        return input_rows
    elif degree > 1:
        final_output=[]
        intermediate_result = polynomial_features_with_Interaction(input_rows, degree-1)
        final_output.extend(intermediate_result)
        for outer_iter in range(len(input_rows)):
            for inner_iter in range(outer_iter,len(intermediate_result)):
                interactions=input_rows[outer_iter]*intermediate_result[inner_iter]
                final_output.append(interactions)
        return final_output

def add_polynomial_features(train_X, degree):
    polynomial_feature_dict = dict()
    if(len(train_X.shape) == 1):
    # we need degree+1 as arange is half open 
        for degree_index in np.arange(1,degree+1):
            polynomial_feature_dict[degree_index] = train_X ** degree_index
        polynomial_feature_list = sorted(polynomial_feature_dict.items(), key=lambda x : x[0])
        polynomial_feature_dict = OrderedDict(polynomial_feature_list)
        train_data = np.column_stack(polynomial_feature_dict.values())
    else:
        for index in range(train_X.shape[0]):
            polynomial_feature_dict[index] = polynomial_features_with_Interaction(train_X[index].tolist(), degree)
        polynomial_feature_list = sorted(polynomial_feature_dict.items(), key=lambda x : x[0])
        polynomial_feature_dict = OrderedDict(polynomial_feature_list)
        train_data = np.vstack(polynomial_feature_dict.values())
    return train_data

def predict(X,W):
    return X.dot(W)

def RMSE_error(observed_y, predicted_y):
    error_numerator = (predicted_y - observed_y)**2
    return np.sqrt(error_numerator.mean())

def k_fold_cross_validation(data, k, labels):
    divided_data = np.array_split(data, k)
    labels = np.array_split(labels, k)
    train_errors = []
    val_errors = []
    for i in range(k):
        #copy to avoid mutation of original data
        data_for_fold = divided_data.copy()
        validation_data = divided_data[i]
        validation_labels = labels[i]
        train_labels = labels.copy()
        del data_for_fold[i]
        del train_labels[i]
        # print((np.concatenate(train_labels).shape))
        print("***********",data_for_fold )
        train_data_for_fold = np.concatenate( data_for_fold, axis=0 )
        weights_estimated = linear_regressor_closed_form(train_data_for_fold, np.concatenate(train_labels))
        predictions_train = predict(train_data_for_fold, weights_estimated)
        train_error = RMSE_error(np.concatenate(train_labels), predictions_train)
        print("train_error in 5 _fold  cross validation", train_error)
        train_errors.append(train_error)
        predictions_val = predict(validation_data, weights_estimated)
        val_error = RMSE_error(validation_labels, predictions_val)
        val_errors.append(val_error)
        print("val_error in 5 fold cross validation", val_error)
    return train_errors, val_errors

    
def train_data_test_data_split(data, labels, split_factor):
    num_train_data = int(split_factor * data.shape[0])
    data_indices = range(data.shape[0])
    data_indices_shuffled = np.random.permutation(data_indices)
    train_data,test_data = data.iloc[data_indices_shuffled[:num_train_data]],data.iloc[data_indices_shuffled[num_train_data:]]
    train_labels, test_labels = labels.iloc[data_indices_shuffled[:num_train_data]], labels.iloc[data_indices_shuffled[num_train_data:]]
    return train_data, test_data, train_labels, test_labels
    
def plot_rmse(degrees, test_mean_rmse):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(degrees, test_mean_rmse)
    ax.set_xlabel('Degree')
    ax.set_ylabel('RMSE')
    plt.show()

def plot_correlation_heatmap(boston_data):
# CRIM - per capita crime rate by town
# ZN - proportion of residential land zoned for lots over 25,000 sq.ft.
# INDUS - proportion of non-retail business acres per town.
# CHAS - Charles River dummy variable (1 if tract bounds river; 0 otherwise)
# NOX - nitric oxides concentration (parts per 10 million)
# RM - average number of rooms per dwelling
# AGE - proportion of owner-occupied units built prior to 1940
# DIS - weighted distances to five Boston employment centres
# RAD - index of accessibility to radial highways
# TAX - full-value property-tax rate per $10,000
# PTRATIO - pupil-teacher ratio by town
# B - 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
# LSTAT - % lower status of the population
# MEDV - Median value of owner-occupied homes in $1000's
    colum_names = { 0:'CRIM',1:'ZN', 2:'INDUS', 3:'CHAS', 4: 'NOX', 5: 'RM', 6: 'AGE', 7: 'DIS', 8: 'RAD', 9: 'TAX', 10: 'PTRATIO', 11:'B', 12: 'LSTAT', 13:'MEDV' }
    boston_data = boston_data.rename(columns = colum_names)
    correlation_matrix = boston_data.corr().round(2)
    sns.heatmap(data=correlation_matrix, annot=True)
    plt.show()

def perform_feature_selection_for_boston_housing(boston_data, train_X, train_Y):
    plot_correlation_heatmap(boston_data)
    return [5, 12]

def prepare_data(features, target):
    train_X, test_X, train_Y, test_Y = train_data_test_data_split(features, target, 0.8)
    print("\n \n")
    print("shape of split data : train data and labels of training dataset: ",train_X.shape, train_Y.shape)

    print("shape of test data and labels", test_X.shape, test_Y.shape)
    train_X_array = train_X.values
    train_Y_array = train_Y.values
    return train_X,train_Y,test_X.values,test_Y.values,train_X_array, train_Y_array

def run_trials(train_X_array, degree, train_Y_array, test_X_array=[],test_Y_array= [], predict_on_test=False):
        X_augmented = add_polynomial_features(train_X_array, degree)
        #insert ones as the first column to consider bias terms
        X_with_bias_terms = np.insert(X_augmented, 0, 1, axis=1)
        print("X_with_bias_terms", X_with_bias_terms[0])
        print("\n\n")
        weights_estimated = linear_regressor_closed_form(X_with_bias_terms, train_Y_array)
        print("shape of weights using closed form linear regressor is",X_with_bias_terms.shape, weights_estimated.shape)
        print("\n\n")
        predictions_train = predict(X_with_bias_terms, weights_estimated)
        train_error = RMSE_error((train_Y_array), predictions_train)
        print("error on 80 percent dataset in case 1 and on entire dataset for question2 ", train_error)
        print("\n\n")

        #predict on test set if flag is true(that is for full 13 features question. For rest it is done in main 
        # method after choosing best degree )
        if predict_on_test:
            test_X_augmented = add_polynomial_features(test_X_array, degree)
            #insert ones as the first column to consider bias terms
            test_X_with_bias_terms = np.insert(test_X_augmented, 0, 1, axis=1)
            predictions_test = predict(test_X_with_bias_terms, weights_estimated)
            test_set_error = RMSE_error(test_Y_array, predictions_test)
            print("test set error when using all 13 features is", test_set_error)
        print("\n\n")

        train_errors, val_errors = k_fold_cross_validation(X_with_bias_terms, 5, train_Y_array)
        print("k_fold_cross_validation mean train error for degree {} and validations errors are: ".format(degree), np.asarray(train_errors).mean(), np.asarray(val_errors).mean())
        train_mean_rmse.append(np.asarray(train_errors).mean())
        test_mean_rmse.append(np.asarray(val_errors).mean())   
        if(len(train_X_array.shape) > 1):
            plt.scatter(train_X_array[:,1], train_Y_array,  color='orange', label = 'labels')  
            plt.plot(train_X_array[:,1], predictions_train, color='green', label='predictions')
            plt.legend()
            plt.xlabel('features')
            plt.ylabel('labels')
            plt.show()
        else:
            plt.scatter(train_X_array, train_Y_array,  color='orange', label='labels')
            #credits i got the sorting idea from a blog post(only the sorting part)
            sorted_zip = sorted(zip(train_X_array,predictions_train), key=operator.itemgetter(0))
            train_X_array, predictions_train = zip(*sorted_zip)  
            #plt.plot(sorted(train_X_array), sorted(predictions_train,reverse = True), color='blue', label='predictions', lw=2)
            plt.plot((train_X_array),(predictions_train), color='green', label='predictions', lw=2)
            plt.legend()
            plt.xlabel('features')
            plt.ylabel('labels')
            plt.show()

        ax1 = sns.distplot(train_Y_array, hist=False, color="r", label="Train_data: Actual Value")
        sns.distplot(predictions_train, hist=False, color="b", label="Train Data: Fitted Values" , ax=ax1)
        plt.show()
        return train_mean_rmse, test_mean_rmse
if __name__ == '__main__':

    boston_data = pd.read_csv('formatted_dataset.data' ,sep=',', header = None, engine='python')
    print("read data", boston_data.head())
    print("\n\n")
    # features and targets
    #MEDV - target
    target = boston_data.iloc[:,13]
    features = boston_data.iloc[:,:13]
    # train_X, test_X, train_Y, test_Y = train_data_test_data_split(features, target, 0.8)
    # print(train_X.shape,test_X.shape, train_Y.shape, test_Y.shape)
    train_X,train_Y,test_X_array,test_Y_array,train_X_array, train_Y_array = prepare_data(features, target)

    # print(train_X[0])
    # polynomial_features= PolynomialFeatures(degree=2)
    # x_poly = polynomial_features.fit_transform(train_X)
    # print("x_poly", x_poly[0])
    degrees = [1,2,3]
    print("\n \n ")
    print("running regression with degrees 1,2,3 for all 13 features(no feature selection")
    print("\n \n")
    train_mean_rmse = []
    test_mean_rmse = []
    for degree in degrees:
        if degree ==1:
            run_trials(train_X_array, degree, train_Y_array, test_X_array,test_Y_array, predict_on_test=True)
        else:
            run_trials(train_X_array, degree, train_Y_array)
    plot_rmse(degrees, train_mean_rmse)
    plot_rmse(degrees, test_mean_rmse)

    perform_feature_selection_for_boston_housing(boston_data, train_X, train_Y)




    #followign code is after just selectign only LSTAT as feature
    print("\n \n ")
    print("running regression with degrees with LSTAT as feature")
    print("\n \n")
    features = boston_data.iloc[:,12]
    print(features.shape)
    _,_,test_X_array,test_Y_array,train_X_array, train_Y_array = prepare_data(features, target)
    degrees = [1,2,3,4,5,6,7]
    train_mean_rmse = []
    test_mean_rmse = []

    min_degree = 0
    min_error = 1e200
    for degree in degrees:
        run_trials(train_X_array, degree, train_Y_array)

    plot_rmse(degrees, train_mean_rmse)
    plot_rmse(degrees, test_mean_rmse)

    min_degree = degrees[test_mean_rmse.index(min(test_mean_rmse))]
    #choose best degree 
    print("min_degree", min_degree)
    print("best degree is {} and  validation error is ".format(min_degree), min(test_mean_rmse))
    X_augmented = add_polynomial_features(train_X_array, min_degree)
    #insert ones as the first column to consider bias terms
    X_with_bias_terms = np.insert(X_augmented, 0, 1, axis=1)
    weights_estimated = linear_regressor_closed_form(X_with_bias_terms, train_Y_array)
    predictions_train = predict(X_with_bias_terms, weights_estimated)
    train_error = RMSE_error((train_Y_array), predictions_train)
    print("\n \n")
    print("RMSE on 80 percent train dataset for best degree is ", train_error)

    test_X_augmented = add_polynomial_features(test_X_array, min_degree)
    #insert ones as the first column to consider bias terms
    test_X_with_bias_terms = np.insert(test_X_augmented, 0, 1, axis=1)
    predictions_test = predict(test_X_with_bias_terms, weights_estimated)
    test_error = RMSE_error((test_Y_array), predictions_test)
    print("\n \n")
    print("RMSE on 20 percent test dataset for best degree is ", test_error)


    print("\n \n ")
    print("running regression with degrees with LSTAT and RM as features")
    print("\n \n")
    # following code s after selecting LSTAT and RM based on heatmap plotted
    features = boston_data.iloc[:,[5,12]]
    print(features.shape)
    _,_,test_X_array,test_Y_array,train_X_array, train_Y_array = prepare_data(features, target)
    degrees = [1,2,3,4,5,6,7]
    train_mean_rmse = []
    test_mean_rmse = []

    min_degree = 0
    min_error = 1e200
    for degree in degrees:
        run_trials(train_X_array, degree, train_Y_array)

    plot_rmse(degrees, train_mean_rmse)
    plot_rmse(degrees, test_mean_rmse)

    new_data = pd.read_csv('data.csv')
    print("\n \n")
    print("new data Question 2 ", new_data.head())
    print("\n \n")
    target = new_data['X']
    features = new_data['Y']
    print("\n \n")
    print("features", np.array2string(features.values, separator=','))
    print("target", np.array2string(target.values, separator=','))
    degrees = [1,2,4,5,10,15,30]
    train_mean_rmse = []
    test_mean_rmse = []
    polynomial_features= PolynomialFeatures(degree=15)
    x_poly = polynomial_features.fit_transform(features.values.reshape(-1, 1))
    print("x_poly", x_poly[0])
    print(features.values.shape)
    for degree in degrees:
        run_trials(features.values, degree, target.values)
    
    plot_rmse(degrees, train_mean_rmse)
    plot_rmse(degrees, test_mean_rmse)