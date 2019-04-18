from keras.applications.mobilenetv2 import MobileNetV2
from keras.layers import Dense, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator

def build_model():
    input_tensor = Input(shape=(224, 224, 3))
    base_model = MobileNetV2(
        include_top=False,
        weights='imagenet',
        input_tensor=input_tensor,
        input_shape=(224, 224, 3),
        pooling='avg')

    for layer in base_model.layers:
        layer.trainable = False
        
    op = Dense(256, activation='relu')(base_model.output)
    op = Dropout(.25)(op)

    output_tensor = Dense(2, activation='sigmoid')(op)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    return model

model = build_model()
model.compile(optimizer=Adam(),
              loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

train_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(224, 224))

validation_generator = train_datagen.flow_from_directory(
    'data/validation',
    target_size=(224, 224))

model.fit_generator(train_generator,
                    steps_per_epoch=train_generator.samples//train_generator.batch_size,
                    validation_data=validation_generator,
                    validation_steps=validation_generator.samples // validation_generator.batch_size,
                    epochs=3)

model.save_weights('second_try.h5')