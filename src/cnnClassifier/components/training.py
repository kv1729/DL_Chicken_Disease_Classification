from cnnClassifier.entity.config_entity import TrainingConfig
import tensorflow as tf
from pathlib import Path


class Training:
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model = None
        self.train_generator = None
        self.valid_generator = None
    
    def get_base_model(self):
        try:
            self.model = tf.keras.models.load_model(self.config.updated_base_model_path)
        except Exception as e:
            print(f"Error loading base model: {e}")
    
    def train_valid_generator(self):
        try:
            datagenerator_kwargs = dict(
                rescale = 1./255,
                validation_split=0.20
            )

            dataflow_kwargs = dict(
                target_size=self.config.params_image_size[:-1],
                batch_size=self.config.params_batch_size,
                interpolation="bilinear"
            )

            valid_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                **datagenerator_kwargs
            )

            self.valid_generator = valid_datagenerator.flow_from_directory(
                directory=self.config.training_data,
                subset="validation",
                shuffle=False,
                **dataflow_kwargs
            )

            if self.config.params_is_augmentation:
                train_datagenerator = tf.keras.preprocessing.image.ImageDataGenerator(
                    rotation_range=40,
                    horizontal_flip=True,
                    width_shift_range=0.2,
                    height_shift_range=0.2,
                    shear_range=0.2,
                    zoom_range=0.2,
                    **datagenerator_kwargs
                )
            else:
                train_datagenerator = valid_datagenerator

            self.train_generator = train_datagenerator.flow_from_directory(
                directory=self.config.training_data,
                subset="training",
                shuffle=True,
                **dataflow_kwargs
            )
        except Exception as e:
            print(f"Error generating data generators: {e}")

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        try:
            model.save(path)
            print(f"Model saved successfully at {path}")
        except Exception as e:
            print(f"Error saving model: {e}")


    def train(self, callback_list: list):
        try:
            if self.model is None:
                print("Model not loaded. Aborting training.")
                return

            if self.train_generator is None or self.valid_generator is None:
                print("Data generators not initialized. Aborting training.")
                return

            self.steps_per_epoch = self.train_generator.samples // self.train_generator.batch_size
            self.validation_steps = self.valid_generator.samples // self.valid_generator.batch_size

            self.model.fit(
                self.train_generator,
                epochs=self.config.params_epochs,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                validation_data=self.valid_generator,
                callbacks=callback_list
            )

            self.save_model(
                path=self.config.trained_model_path,
                model=self.model
            )
        except Exception as e:
            print(f"Error during training: {e}")
