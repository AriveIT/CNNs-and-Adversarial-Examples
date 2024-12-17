import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from scipy import stats # for mode calculation

# Softmax temperature layer
class SoftmaxTemperature(tf.keras.layers.Layer):
    def __init__(self, temperature=1.0, **kwargs):
        super(SoftmaxTemperature, self).__init__(**kwargs)
        self.temperature = temperature

    def call(self, inputs):
        logits = inputs / self.temperature
        return tf.nn.softmax(logits)

# This class is essentially just to keep notebooks clean
class AdversarialExamples:
    def one_model_ensemble_predict(model, image, eps, n):
        perturbed_images = np.empty((n, *image.shape))
        for i in range(n):
            perturbed_images[i] = image + eps * AdversarialExamples.create_random_pattern(image.shape)
        predictions = model(perturbed_images)
        pred_indices = np.argmax(predictions, axis=1)
        return stats.mode(pred_indices, keepdims=False)[0]
    
    @staticmethod
    def augment(image, label):
    
        # Apply random augmentations
        image = tf.image.random_flip_left_right(image)  # Random horizontal flip
        image = tf.image.random_brightness(image, max_delta=0.1)  # Adjust brightness
        image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Adjust contrast
        image = tf.pad(image, paddings=[[2, 2], [2, 2], [0, 0]], mode='REFLECT') # reflect to remove black borders from cropping
        image = tf.image.random_crop(image, size=[32, 32, 3])  # Crop back to 32x32
        image = tf.clip_by_value(image, 0.0, 1.0) # clip values to within valid range

        return image, label

    # x is data, y is one-hot labels
    @staticmethod
    def create_fgsm_dataset(model, x, y, eps):
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        x_adv = np.empty(x.shape)

        with tf.GradientTape() as tape:
            tape.watch(x)
            predictions = model(x)
            loss = tf.keras.losses.categorical_crossentropy(y, predictions)
        
        gradients = tape.gradient(loss, x)
        x_adv = x + eps * tf.sign(gradients)
        x_adv = np.clip(x_adv, 0, 1)

        return x_adv

    @staticmethod
    def create_random_pattern(shape):
        pattern = np.random.randint(0, 2, size=shape)
        pattern = pattern * 2 - 1 # change [0,1] to [-1, 1]
        # print(pattern)
        return pattern

    @staticmethod
    def create_fgsm_pattern(model, input_image, input_label):
        loss_object = tf.keras.losses.CategoricalCrossentropy()
        input_image_tensor = tf.convert_to_tensor(input_image, dtype=tf.float32)
        input_image_tensor = tf.expand_dims(input_image_tensor, axis=0) # batch size of 1
        input_label = np.expand_dims(input_label, axis=0) # batch size of 1

        with tf.GradientTape() as tape:
            tape.watch(input_image_tensor)
            prediction = model(input_image_tensor)
            loss = loss_object(input_label, prediction)
        
        gradient = tape.gradient(loss, input_image_tensor) # gradient of loss wrt input image
        signed_grad = tf.sign(gradient) # sign of the gradients
        return signed_grad

    # returns untrained model
    @staticmethod
    def get_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10, activation='softmax')
        ])

        model.build()
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        model.compile(optimizer='adam',
                    loss=loss_fn,
                    metrics=['accuracy'])

        return model
    
    @staticmethod
    def get_paper_model(temperature):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),

            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.5),

            tf.keras.layers.Dense(10),
            SoftmaxTemperature(temperature=temperature)

        ])

        model.build()
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        model.compile(optimizer='adam',
                    loss=loss_fn,
                    metrics=['accuracy'])

        return model

    @staticmethod
    def get_distillation_model(temperature):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3), padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.3),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(10),
            SoftmaxTemperature(temperature=temperature)
        ])

        model.build()
        loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
        model.compile(optimizer='adam',
                loss=loss_fn,
                metrics=['accuracy'])

        return model

    def test_adversarial_dataset(model, x, y, eps, n, model_name):
        x_adv_test = AdversarialExamples.create_fgsm_dataset(model, x[:n], y[:n], eps=eps)
        loss, accuracy = model.evaluate(x_adv_test, y[:n], verbose=1)

        predictions = tf.argmax(model(x_adv_test), axis=1)
        conf_mat = tf.math.confusion_matrix(y[:n], predictions)

        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_mat.numpy(), annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title(f"{model_name} Confusion Matrix")
        plt.show()