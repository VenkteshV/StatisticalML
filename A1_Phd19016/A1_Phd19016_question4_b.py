import numpy as np
def discriminant_fun(feature_vector, mean,variance, prior):
    #ignoring unimportant additive constants 
    # params feature_vector : x a d * 1 dim array
    # mean d * 1 dim array of mean
    #prior(unless all priors are equal this must be included)
    g = (0.5 * (1/(variance **2)) * mean.T).dot(feature_vector)
    - (1/(2*(variance **2))* ((mean).T).dot(mean))
    + np.log(prior)
    return float(g)


if __name__ == '__main__':
    rows_mean = int(input("Enter the number of rows of mean")) 
    columns_mean = int(input("Enter the number of columns of mean "))
    print("please enter mean entries  in one line (separated by space):  ")
    entries = list(map(int, input().split())) 
    mean = np.array(entries).reshape(rows_mean,columns_mean) 
    rows_x = int(input("Enter the number of rows of feature_matrix")) 
    columns_x = int(input("Enter the number of columns of feature_matrix "))
    print("please enter feture matrix entries in one line (separated by space):  ")
    x_vec = list(map(int, input().split())) 
    x_vector = np.array(x_vec).reshape(rows_x, columns_x)
    prior = float(input("enter prior probability"))
    variance = float(input("enter variance values"))
    discriminant_value = discriminant_fun(x_vector, mean,variance, prior ) 
    print(discriminant_value)