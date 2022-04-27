import pickle
import metrics


def evaluate(model_name,isOverSampled=False):
    import utils
    data_type="oversampled" if isOverSampled else "standard"
    if "AlexNet" in model_name:
        x_train,x_test,y_train,y_test=utils.load_standard_data(data_type=data_type,isAlexnet=True)
    else:
        x_train,x_test,y_train,y_test=utils.load_standard_data(data_type=data_type)
    # Load the Model back from file
    if "AlexNet" in model_name:
        import tensorflow as tf
        from tensorflow import keras
        model = keras.models.Sequential([
            keras.layers.Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=(227,227,1)),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=384, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(1,1), strides=(1,1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.MaxPool2D(pool_size=(3,3), strides=(2,2)),
            keras.layers.Flatten(),
            keras.layers.Dense(9216,input_shape=(12544,), activation='relu'),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(4096, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(1, activation='sigmoid')
            ])
        from keras import backend as K
        def recall_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
            recall = true_positives / (possible_positives + K.epsilon())
            return recall

        def precision_m(y_true, y_pred):
            true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
            predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
            precision = true_positives / (predicted_positives + K.epsilon())
            return precision

        def f1_m(y_true, y_pred):
            precision = precision_m(y_true, y_pred)
            recall = recall_m(y_true, y_pred)
            return 2*((precision*recall)/(precision+recall+K.epsilon()))
        model.compile(
            optimizer=tf.optimizers.Adam(lr=0.000001),
            loss='binary_crossentropy',
            metrics=['accuracy','Recall','AUC','FalseNegatives',f1_m]
        )
        model.load_weights(f'../models/{model_name}.hdf5')
        print("Model: ",model_name)
        print("Training data")
        model.evaluate(x_train,y_train)
        print("Test data")
        model.evaluate(x_test,y_test)
        return

    with open(f'../models/{model_name}.model', 'rb') as file:  
        model = pickle.load(file)

    # if "SVM" in model_name:
    #     print("Model: ",model_name)
    #     print("Training data")
    #     y_pred_train = model.predict(x_train)
    #     print(model.score(x_train,y_train))
    #     metrics.conf_matrix(y_train,y_pred_train)
    #     print("Test data")
    #     print(model.score(x_test,y_test))
    #     y_pred_test = model.predict(x_test)
    #     metrics.conf_matrix(y_test,y_pred_test)
    #     return

    print("Model: ",model_name)
    sc_train = model.score(x_train, y_train)
    sc_test = model.score(x_test, y_test)
    print("Train data")
    y_pred_train = model.predict(x_train)
    probs=model.predict_proba(x_train)
    print(sc_train)
    metrics.conf_matrix(y_train,y_pred_train)
    metrics.roc_pr_curve(y_train,probs[:,1])
    print("Test data")
    y_pred_test = model.predict(x_test)
    probs=model.predict_proba(x_test)
    print(sc_test)

    metrics.conf_matrix(y_test,y_pred_test)
    metrics.roc_pr_curve(y_test,probs[:,1])
    
    del x_train, x_test, y_train, y_test
    



