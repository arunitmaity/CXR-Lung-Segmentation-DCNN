from tensorflow.keras import layers
from tensorflow.keras import backend
from tensorflow.keras.models import load_model
from glob import glob
import os
import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
import tensorflow.keras.backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import cv2


class FixedDropout(layers.Dropout):
    def _get_noise_shape(self, inputs):
        if self.noise_shape is None:
            return self.noise_shape
        symbolic_shape = backend.shape(inputs)
        noise_shape = [symbolic_shape[axis] if shape is None else shape
                       for axis, shape in enumerate(self.noise_shape)]
        return tuple(noise_shape)


smooth = 1e-6


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)


def jacard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def generalised_dice_loss_2d(Y_gt, Y_pred):
    w = tf.reduce_sum(Y_gt)
    w = 1 / (w ** 2 + smooth)

    numerator = Y_gt * Y_pred
    numerator = w * tf.reduce_sum(numerator)
    numerator = tf.reduce_sum(numerator)

    denominator = Y_pred + Y_gt
    denominator = w * tf.reduce_sum(denominator)
    denominator = tf.reduce_sum(denominator)

    gen_dice_coef = 2 * numerator / (denominator + smooth)
    loss = tf.reduce_mean(1 - gen_dice_coef)
    return loss


def pgdl(y_true, y_pred):
    k = 2.5
    gd = generalised_dice_loss_2d(y_true, y_pred)
    loss = gd / (1 + k * (1 - gd))
    return loss


alpha = 0.9


def loss(y_true, y_pred):
    return alpha * pgdl(y_true, y_pred) + (1 - alpha) * binary_crossentropy(y_true, y_pred)


model = load_model('net_lung_seg.hdf5',
                   custom_objects={"FixedDropout": FixedDropout, "loss": loss, "dice_coef": dice_coef, "pgdl": pgdl,
                                   "generalised_dice_loss_2d": generalised_dice_loss_2d, "jacard_coef": jacard_coef,
                                   "dice_coef_loss": dice_coef_loss})

files = []


def cxr_generator(batch_size, image_path, image_folder, files,
                  image_color_mode="rgb",
                  target_size=(512, 512)):
    image_datagen = ImageDataGenerator()
    image_generator = image_datagen.flow_from_directory(
        image_path,
        classes=[image_folder],
        class_mode=None,
        color_mode=image_color_mode,
        target_size=target_size,
        batch_size=batch_size,
        shuffle=False)
    image_generator.reset()
    files.extend(image_generator.filenames)
    for img in image_generator:
        img = img / 255.0
        yield img


BATCH_SIZE = 1
segmentation_train_path = '/home/arunitmaity/Projects/Lung_Segmentation/processed_data/tbh+clahe/train/'
segmentation_validation_path = '/home/arunitmaity/Projects/Lung_Segmentation/processed_data/tbh+clahe/validation/'
segmentation_test_path = '/home/arunitmaity/Projects/Lung_Segmentation/processed_data/tbh+clahe/test/'
segmentation_validation_image_path = '/home/arunitmaity/Projects/Lung_Segmentation/processed_data/tbh+clahe/validation/cxrs/'
segmentation_train_image_path = '/home/arunitmaity/Projects/Lung_Segmentation/processed_data/tbh+clahe/train/cxrs/'
segmentation_test_image_path = '/home/arunitmaity/Projects/Lung_Segmentation/processed_data/tbh+clahe/test/cxrs/'
train_files = glob(os.path.join(segmentation_train_image_path, "*.png"))
val_files = glob(os.path.join(segmentation_validation_image_path, "*.png"))
test_files = glob(os.path.join(segmentation_test_image_path, "*.png"))

# train_gen = cxr_generator(BATCH_SIZE,
#                           segmentation_train_path,
#                           'cxrs',
#                           target_size=(512, 512))
#
# val_gen = cxr_generator(BATCH_SIZE,
#                         segmentation_validation_path,
#                         'cxrs',
#                         target_size=(512, 512))

test_gen = cxr_generator(BATCH_SIZE,
                         segmentation_test_path,
                         'cxrs',
                         files=files,
                         target_size=(512, 512))

results = model.predict(test_gen, verbose=1, steps=len(test_files) / BATCH_SIZE)


def save_result(save_path, npyfile, files):
    for i, item in enumerate(npyfile):
        result_file = files[i]
        item[item >= 0.50] = 1.0
        item[item < 0.50] = 0.0
        img = (item[:, :, 0] * 255.).astype(np.uint8)
        filename = os.path.basename(result_file)
        result_file = os.path.join(save_path, filename)
        cv2.imwrite(result_file, img)


save_result('/home/arunitmaity/Projects/Lung_Segmentation/results/predicted_masks', results, files)
