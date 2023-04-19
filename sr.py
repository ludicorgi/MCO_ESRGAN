import os
from PIL import Image
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
os.environ["TFHUB_DOWNLOAD_PROGRESS"] = "True"

# Code reference: https://www.tensorflow.org/hub/tutorials/image_enhancing
# Declaring Constants
IMAGE_PATH = "test.jpg"
SAVED_MODEL_PATH = "https://tfhub.dev/captain-pool/esrgan-tf2/1"

def preprocess_image(image):
  """ Loads image from path and preprocesses to make it model ready
      Args:
        image_path: Path to the image file
  """
  hr_image = tf.keras.utils.img_to_array(image)
  # hr_image = tf.image.decode_image(tf.io.read_file(image_path))

  # If PNG, remove the alpha channel. The model only supports
  # images with 3 color channels.
  if hr_image.shape[-1] == 4:
    hr_image = hr_image[...,:-1]
  hr_size = (tf.convert_to_tensor(hr_image.shape[:-1]) // 4) * 4
  hr_image = tf.image.crop_to_bounding_box(hr_image, 0, 0, hr_size[0], hr_size[1])
  hr_image = tf.cast(hr_image, tf.float32)
  return tf.expand_dims(hr_image, 0)

def save_image(image, filename):
  """
    Saves unscaled Tensor Images.
    Args:
      image: 3D image tensor. [height, width, channels]
      filename: Name of the file to save.
  """
  if not isinstance(image, Image.Image):
    image = tf.clip_by_value(image, 0, 255)
    image = Image.fromarray(tf.cast(image, tf.uint8).numpy())
  image.save("%s.png" % filename)
  print("Saved as %s.png" % filename)

def downscale_image(image):
  """
      Scales down images using bicubic downsampling.
      Args:
          image: 3D or 4D tensor of preprocessed image
  """
  image_size = []
  if len(image.shape) == 3:
    image_size = [image.shape[1], image.shape[0]]
  else:
    raise ValueError("Dimension mismatch. Can work only on single image.")

  image = tf.squeeze(
      tf.cast(
          tf.clip_by_value(image, 0, 255), tf.uint8))

  lr_image = np.asarray(
    Image.fromarray(image.numpy())
    .resize([image_size[0] // 4, image_size[1] // 4],
              Image.BICUBIC))

  lr_image = tf.expand_dims(lr_image, 0)
  lr_image = tf.cast(lr_image, tf.float32)
  return lr_image

def upscale(image):
  hr_image = preprocess_image(image)
  lr_image = downscale_image(tf.squeeze(hr_image))

  model = hub.load(SAVED_MODEL_PATH)

  sr_image = model(lr_image)
  sr_image = tf.squeeze(sr_image)

  # save_image(sr_image, "output")

  print("Image successfully upscaled")

  return sr_image, tf.squeeze(lr_image)

# if __name__ == '__main__':
#   hr_image = preprocess_image(IMAGE_PATH)
#   lr_image = downscale_image(tf.squeeze(hr_image))

#   model = hub.load(SAVED_MODEL_PATH)

#   fake_image = model(lr_image)
#   fake_image = tf.squeeze(fake_image)

#   save_image(fake_image, "output_lr")
#   pass

