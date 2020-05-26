for layer in ResNet_model.layers[:-15]:
    layer.trainable = False

x = ResNet_model.output
x = GlobalAveragePooling2D()(x)
x = Flatten()(x)
x = Dense(units=512, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(units=700, activation='relu')(x)
x = Dropout(0.5)(x)
output  = Dense(units=2, activation='relu')(x)
model = Model(ResNet_model.input, output)
