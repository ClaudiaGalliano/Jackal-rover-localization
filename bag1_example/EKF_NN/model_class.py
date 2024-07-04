from sklearn.metrics import mean_squared_error
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.models import Sequential, load_model
import tensorflow.python.keras.optimizers as optim
import numpy as np


def create_model(X_train, activation='relu', units1 = 10, units2 = 15, lr = 0.001):
    model = Sequential()
    model.add(Dense(units=len(X_train[0]), activation='relu',input_dim = len(X_train[0])))
    model.add(Dense(units=units1, activation='relu')) 
    model.add(Dense(units=units2, activation='relu')) 
    model.add(Dense(units=units1, activation='relu')) 
    model.add(Dense(units=len(X_train[0]), activation=activation))
    opt = optim.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mean_squared_error',metrics=['accuracy']) 
    return model

class AE_RNG:
    def __init__(self, model_path = None, train=True):
        if train:
            self.model = None
        else:    
            self.model = load_model(model_path)
        self.prediction = None
    
    def train(self, X_train, y_train, epochs=150, batch_size=32):
        activation = 'leaky_relu'
        units1 = 15
        units2 = 30
        lr = 0.001
        batch_size = 32
        epochs = 150
        self.model = create_model(X_train,activation,units1,units2,lr)
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=2, shuffle=True)
        self.model.save('AE_RNG_Model.h5')
    
    def predict(self, data):
        if data.shape == (4,):
            self.prediction = self.model.predict(data.reshape(1,-1))
        else:
            self.prediction = self.model.predict(data)
        return self.prediction
    
    def score(self, data):
        if data.shape == (4,):
            self.prediction = self.model.predict(data.reshape(1,-1))
        else:
            self.prediction = self.model.predict(data)
        error = np.absolute(self.prediction - data)
        absolute_error = np.sqrt(np.square(error[:,0]) + np.square(error[:,1]) + np.square(error[:,2]) + np.square(error[:,3]))
        return error, absolute_error
    
    def test(self, X_test, y_test,X_train,y_train):
        error1train, error2train, error3train, error4train = [],[],[],[]
        error1test, error2test, error3test, error4test = [],[],[],[]

        y_pred_test = self.predict(X_test)
        y_pred_train = self.predict(X_train)

        mse_test = mean_squared_error(y_test, y_pred_test)
        mse_train = mean_squared_error(y_train, y_pred_train)

        error_test = np.absolute(y_pred_test - X_test)
        error_train = np.absolute(y_pred_train - X_train)
        

        for elm in error_test:
            error1test.append(elm[0])
            error2test.append(elm[1])
            error3test.append(elm[2])
            error4test.append(elm[3])


        for elm in error_train:
            error1train.append(elm[0])
            error2train.append(elm[1])
            error3train.append(elm[2])
            error4train.append(elm[3])
        
        z_test = np.sqrt(np.square(error1test) + np.square(error2test) + np.square(error3test) + np.square(error4test))
        z_train = np.sqrt(np.square(error1train) + np.square(error2train) + np.square(error3train) + np.square(error4train))

        return y_pred_test, y_pred_train, z_test, z_train, mse_test, mse_train, error1test, error2test, error3test, error4test, error1train, error2train, error3train, error4train
    