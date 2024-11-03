from sklearn.model_selection import train_test_split


Xtrain, Xval, Ytrain, Yval = train_test_split(Xtrain, Ytrain, test_size=0.25, random_state=42)

# Print the size of training data, validation data and test data
print("Size of Train data is: {} and Size of Train labels is: {}".format(Xtrain.shape, Ytrain.shape))
print("Size of Validation data is: {} and Size of Validation labels is: {}".format(Xval.shape, Yval.shape))
print("Size of Test data is: {} and Size of Test labels is: {}".format(Xtest.shape, Ytest.shape))
