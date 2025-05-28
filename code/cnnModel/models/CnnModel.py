# This contains the code for designing the CNN model

from tensorflow.keras import layers, models

def build_cnn(input_shape, num_classes=4):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv1D(64, 15, activation='relu', padding='same'), 
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(128, 7, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Conv1D(256, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
        
        
        # Low complexity model
        
        # layers.Input(shape=input_shape),  
        # # Reduced filters and kernel size
        # layers.Conv1D(32, 7, activation='relu', padding='same'),
        # layers.BatchNormalization(),
        # layers.MaxPooling1D(2),
        # layers.Conv1D(64, 5, activation='relu', padding='same'),
        # layers.BatchNormalization(),
        # layers.MaxPooling1D(2),
        # layers.GlobalAveragePooling1D(),
        # # Smaller dense layer
        # layers.Dense(64, activation='relu'),
        # layers.Dropout(0.3),
        # layers.Dense(num_classes, activation='softmax')
        
        
        # Previous research cnn model
        
        # layers.Conv1D(128, 55, activation='relu', padding='same'),
        # layers.MaxPooling1D(5),  # reduced from 10 to 5
        # layers.Dropout(0.5),
        # layers.Conv1D(128, 25, activation='relu', padding='same'),
        # layers.MaxPooling1D(3),  # reduced from 5 to 3
        # layers.Dropout(0.5),
        # layers.Conv1D(128, 10, activation='relu', padding='same'),
        # layers.MaxPooling1D(3),  # reduced from 5 to 3
        # layers.Dropout(0.5),
        # layers.Conv1D(128, 5, activation='relu', padding='same'),
        # layers.GlobalAveragePooling1D(),
        # layers.Dense(256, kernel_initializer='normal', activation='relu'),
        # layers.Dropout(0.5),
        # layers.Dense(128, kernel_initializer='normal', activation='relu'),
        # layers.Dropout(0.5),
        # layers.Dense(64, kernel_initializer='normal', activation='relu'),
        # layers.Dropout(0.5),
        # layers.Dense(num_classes, kernel_initializer='normal', activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model


# Example usage
# input_shape = (250,1)
# model = build_cnn(input_shape, num_classes=5)
# model.summary()

