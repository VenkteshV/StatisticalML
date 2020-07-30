import numpy as np
import matplotlib.pyplot as plt

def generate_random_variables(mean, covariance_matrix, size = 2000):
     # checks could be made to enforce that both are matrics , 
     # that is d >1 but since it is not specified and d could also be 1 
     # this method is open ended
     return np.random.multivariate_normal(mean, covariance_matrix, size)



if __name__ == '__main__':
    shape = int(input("Enter the number of entries of mean")) 
    print("please enter mean entries  in one line (separated by space):  ")
    entries = list(map(int, input().split())) 
    mean = np.array(entries).reshape(shape) 
    print(mean)
    rows_cov = int(input("Enter the number of rows of covariance_matrix")) 
    columns_cov = int(input("Enter the number of columns of covariance_matrix ")) 
    print("please enter covariance matrix entries in one line (separated by space):  ")
    cov = list(map(int, input().split())) 
    covariance_matrix = np.array(cov).reshape(rows_cov, columns_cov) 
    print(covariance_matrix)
    x = generate_random_variables(mean, covariance_matrix)
    print("generated output is : ", x)
    print("shape of output : ", x.shape)
    x_vector, y_vector = x.T
    plt.plot(x_vector, y_vector, 'x')
    plt.axis('equal')
    plt.show()