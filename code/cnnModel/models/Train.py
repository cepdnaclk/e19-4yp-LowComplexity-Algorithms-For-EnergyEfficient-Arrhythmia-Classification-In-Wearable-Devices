import tensorflow as tf
import numpy as np
from .CnnModel import build_cnn
from .CnnModelWithChannelAttention import build_cnn_with_attention

def train_model(X_train, y_train, X_val, y_val):
    # model = build_cnn(input_shape=X_train.shape[1:])
    model = build_cnn_with_attention(input_shape=X_train.shape[1:], num_classes=len(np.unique(y_train)))
    
    # Compile the model explicitly
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Define early stopping according to the improvement of val_loss (Validation loss)
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks
    )
    
    return model, history

