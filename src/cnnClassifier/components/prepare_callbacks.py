import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
import time
from cnnClassifier.entity.config_entity import PrepareCallbacksConfig

class PrepareCallback:
    def __init__(self, config: PrepareCallbacksConfig):
        self.config = config

    def _create_tb_callbacks(self):
        try:
            timestamp = time.strftime("%Y-%m-%d-%H-%M-%S")
            tb_running_log_dir = os.path.join(
                self.config.tensorboard_root_log_dir,
                f"tb_logs_at_{timestamp}",
            )
            return tf.keras.callbacks.TensorBoard(log_dir=tb_running_log_dir)
        except Exception as e:
            print(f"Error creating TensorBoard callback: {e}")
            return None

    def _create_ckpt_callbacks(self):
        try:
            return tf.keras.callbacks.ModelCheckpoint(
                filepath=self.config.checkpoint_model_filepath.with_suffix('.keras'), # Append .keras extension
                save_best_only=True
            )
        except Exception as e:
            print(f"Error creating ModelCheckpoint callback: {e}")
            return None

    def get_tb_ckpt_callbacks(self):
        tb_callback = self._create_tb_callbacks()
        ckpt_callback = self._create_ckpt_callbacks()
        if tb_callback is not None and ckpt_callback is not None:
            return [tb_callback, ckpt_callback]
        else:
            return []
