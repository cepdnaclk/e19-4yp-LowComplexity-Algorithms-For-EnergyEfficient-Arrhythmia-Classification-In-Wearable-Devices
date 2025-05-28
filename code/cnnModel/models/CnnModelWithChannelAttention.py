from tensorflow.keras import layers, models

def channel_attention(input_feature, ratio=8):
    """Channel Attention Module for 1D signals"""
    channel = input_feature.shape[-1]
    
    # Average and Max Pooling
    avg_pool = layers.GlobalAveragePooling1D()(input_feature)
    max_pool = layers.GlobalMaxPooling1D()(input_feature)
    
    # Shared MLP with bottleneck
    dense1 = layers.Dense(channel//ratio, activation='relu')
    dense2 = layers.Dense(channel)
    
    avg_out = dense2(dense1(avg_pool))
    max_out = dense2(dense1(max_pool))
    
    # Merge and activate
    channel_weights = layers.Activation('sigmoid')(layers.Add()([avg_out, max_out]))
    channel_weights = layers.Reshape((1, channel))(channel_weights)
    
    return layers.Multiply()([input_feature, channel_weights])

def build_cnn_with_attention(input_shape, num_classes=4):
    """Modified CNN with Channel Attention Modules"""
    inputs = layers.Input(shape=input_shape)
    
    # Block 1
    x = layers.Conv1D(64, 15, padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = channel_attention(x)  # Attention after first conv
    x = layers.MaxPooling1D(2)(x)
    
    # Block 2
    x = layers.Conv1D(128, 7, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = channel_attention(x)  # Attention after second conv
    x = layers.MaxPooling1D(2)(x)
    
    # Block 3
    x = layers.Conv1D(256, 5, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = channel_attention(x)  # Attention after third conv
    
    # Final layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    return models.Model(inputs=inputs, outputs=outputs)
