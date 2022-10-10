import tensorflow as tf
from tensorflow import keras
from model import make_model


def main():
    image_size = (256, 256)
    batch_size = 2
    epochs = 50

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data",
        validation_split=0.2,
        subset="training",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )
    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        "data",
        validation_split=0.2,
        subset="validation",
        seed=1337,
        image_size=image_size,
        batch_size=batch_size,
    )

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)

    model = make_model(input_shape=image_size + (3,), num_classes=2)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint
        (
            "model.h5",
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            save_freq='epoch'
        )
    ]
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.fit(
        train_ds, epochs=epochs, callbacks=callbacks, validation_data=val_ds,
    )

    img = keras.preprocessing.image.load_img(
        "data/cringe/Ash.png", target_size=image_size
    )
    classify_image(img)


def classify_image(img):
    model = tf.keras.models.load_model("model.h5")
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    predictions = model.predict(img_array)
    score = predictions[0]
    print(
        "This image is %.2f percent cringe and %.2f percent pog."
        % (100 * (1 - score), 100 * score)
    )
    return 100 * (1 - score)  # cringe factor


if __name__ == "__main__":
    main()