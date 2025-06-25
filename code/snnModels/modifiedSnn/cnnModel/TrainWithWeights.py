import tensorflow as tf
import numpy as np 
from .CnnModel import build_cnn

def train_model_with_weights(X_train, y_train, X_val, y_val, class_weights=None):
    model = build_cnn(input_shape=X_train.shape[1:], num_classes=len(np.unique(y_train)))
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=100,
        callbacks=callbacks,
        class_weight=class_weights  
    )
    
    return model, history
