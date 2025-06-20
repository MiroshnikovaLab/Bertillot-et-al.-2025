import numpy as np
import tensorflow as tf
from keras.models import Model
from tensorflow.keras import backend as keras
from math import ceil
import numpy as np
from skimage import io

def weighted_binary_crossentropy(class_weights: np.ndarray):
    """
    Returns a function that calculates the weighted binary cross-entropy
    given the provided class weights.

    Args:
        class_weights (np.ndarray): A length-2 array-like containing the 
            weights for the negative (class_weights[0]) and positive 
            (class_weights[1]) classes respectively.

    Returns:
        Callable: A function that takes y_true and y_pred as inputs and 
        returns the mean weighted binary cross-entropy.
    """

    def _weighted_binary_crossentropy(y_true: np.ndarray, y_pred: np.ndarray) ->  tf.Tensor:
        """
        Calculates the mean weighted binary cross-entropy for the given 
        true labels (y_true) and predicted labels (y_pred).

        Args:
            y_true (np.ndarray): Ground-truth binary labels.
            y_pred (np.ndarray): Predicted probabilities (output of a 
                sigmoid for binary classification).

        Returns:
            keras.Tensor: The mean weighted binary cross-entropy.
        """
        binary_crossentropy = keras.backend.binary_crossentropy(y_true, y_pred)
        weight_vector = y_true * class_weights[1] + (1.0 - y_true) * class_weights[0]
        weighted_binary_crossentropy = weight_vector * binary_crossentropy

        return keras.backend.mean(weighted_binary_crossentropy)

    return _weighted_binary_crossentropy


def normalize_mi_ma(
    x: np.ndarray,
    mi: float,
    ma: float,
    clip: bool = False,
    eps: float = 1e-20,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Normalizes the array `x` to the range [0, 1] based on the given minimum 
    (`mi`) and maximum (`ma`) values, adding a small epsilon (`eps`) to 
    avoid division by zero.

    This function is adapted from Martin Weigert.

    Args:
        x (np.ndarray): Input array to be normalized.
        mi (float): Lower bound for normalization. Could be a scalar or 
            computed percentile.
        ma (float): Upper bound for normalization. Could be a scalar or 
            computed percentile.
        clip (bool, optional): If True, clips the result to [0, 1]. 
            Defaults to False.
        eps (float, optional): A small constant to avoid division by zero. 
            Defaults to 1e-20.
        dtype (np.dtype, optional): Data type to cast the array and 
            bounds. Defaults to np.float32.

    Returns:
        np.ndarray: The normalized array.
    """
    if dtype is not None:
        x = x.astype(dtype, copy=False)
        mi = dtype(mi) if np.isscalar(mi) else mi.astype(dtype, copy=False)
        ma = dtype(ma) if np.isscalar(ma) else ma.astype(dtype, copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / (ma - mi + eps)")
    except ImportError:
        x = (x - mi) / (ma - mi + eps)

    if clip:
        x = np.clip(x, 0, 1)

    return x


def normalizePercentile(
    x: np.ndarray,
    pmin: float = 1,
    pmax: float = 99.8,
    axis=None,
    clip: bool = False,
    eps: float = 1e-20,
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Percentile-based image normalization. Normalizes `x` so that the 
    pmin-th and pmax-th percentiles of `x` become 0 and 1 respectively.

    This function is adapted from Martin Weigert.

    Args:
        x (np.ndarray): Input array to be normalized.
        pmin (float, optional): Lower percentile. Defaults to 1.
        pmax (float, optional): Upper percentile. Defaults to 99.8.
        axis (int or tuple of int, optional): Axis/axes along which 
            the percentiles are computed. Defaults to None.
        clip (bool, optional): If True, clips the result to [0, 1]. 
            Defaults to False.
        eps (float, optional): A small constant to avoid division by zero. 
            Defaults to 1e-20.
        dtype (np.dtype, optional): Data type to cast the array. 
            Defaults to np.float32.

    Returns:
        np.ndarray: The percentile-normalized array.
    """
    mi = np.percentile(x, pmin, axis=axis, keepdims=True)
    ma = np.percentile(x, pmax, axis=axis, keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)


def predict_prob_as_tiles(image_path: str,
                            model: Model) -> np.ndarray:
    """
    Predict with 50% tile overlap and smooth Hann‐window blending.
    """
    # Load & normalize
    img0 = io.imread(image_path, as_gray=True)
    img  = normalizePercentile(img0)
    H, W = img.shape

    # Tile size & half‐overlap stride
    ph, pw = model.input_shape[1:3]
    stride_h, stride_w = ph // 2, pw // 2

    # Reflect padding by half‐tile
    pad_h, pad_w = ph//2, pw//2
    img_p = np.pad(img, ((pad_h,pad_h), (pad_w,pad_w)), mode='reflect')

    # Build 2D Hann window
    wy = np.hanning(ph)
    wx = np.hanning(pw)
    window = np.outer(wy, wx)

    # Accumulators
    prob_accum   = np.zeros_like(img_p, dtype=np.float32)
    weight_accum = np.zeros_like(img_p, dtype=np.float32)

    # Slide & accumulate
    for y0 in range(0, W + 2*pad_w - pw + 1, stride_w):
        for x0 in range(0, H + 2*pad_h - ph + 1, stride_h):
            patch = img_p[x0:x0+ph, y0:y0+pw]
            inp   = patch[np.newaxis, ..., np.newaxis]  # (1,ph,pw,1)
            probs = model.predict(inp, batch_size=1, verbose=0)[0]  # (ph,pw,2)
            p1    = probs[..., 1]

            prob_accum  [x0:x0+ph, y0:y0+pw] += p1 * window
            weight_accum[x0:x0+ph, y0:y0+pw] += window

    # Normalize, crop back to original, threshold
    avg_prob = prob_accum / (weight_accum + 1e-12)
    avg_prob = avg_prob[pad_h:pad_h+H, pad_w:pad_w+W]

    return avg_prob

def predict_as_tiles(Image_path, model):

  # Read the data in and normalize
  image_raw = io.imread(Image_path, as_gray = True)
  print(image_raw.shape)
  image_raw = normalizePercentile(image_raw)

  # Get the patch size from the input layer of the model
  # patch_size = model.layers[0].output_shape[0][1:3]
  patch_size = model.input_shape[1:3]

  # Pad the image with zeros if any of its dimensions is smaller than the patch size
  if image_raw.shape[0] < patch_size[0] or image_raw.shape[1] < patch_size[1]:
    Image = np.zeros((max(image_raw.shape[0], patch_size[0]), max(image_raw.shape[1], patch_size[1])))
    Image[0:image_raw.shape[0], 0: image_raw.shape[1]] = image_raw
  else:
    Image = image_raw

  # Calculate the number of patches in each dimension
  n_patch_in_width = ceil(Image.shape[0]/patch_size[0])
  n_patch_in_height = ceil(Image.shape[1]/patch_size[1])

  prediction = np.zeros(Image.shape, dtype = 'uint8')

  for x in range(n_patch_in_width):
    for y in range(n_patch_in_height):
      xi = patch_size[0]*x
      yi = patch_size[1]*y

      # If the patch exceeds the edge of the image shift it back
      if xi+patch_size[0] >= Image.shape[0]:
        xi = Image.shape[0]-patch_size[0]

      if yi+patch_size[1] >= Image.shape[1]:
        yi = Image.shape[1]-patch_size[1]

      # Extract and reshape the patch
      patch = Image[xi:xi+patch_size[0], yi:yi+patch_size[1]]
      patch = np.reshape(patch,patch.shape+(1,))
      patch = np.reshape(patch,(1,)+patch.shape)

      # Get the prediction from the patch and paste it in the prediction in the right place
      predicted_patch = model.predict(patch, batch_size = 1)
      prediction[xi:xi+patch_size[0], yi:yi+patch_size[1]] = (np.argmax(np.squeeze(predicted_patch), axis = -1)).astype(np.uint8)


  return prediction[0:image_raw.shape[0], 0: image_raw.shape[1]]

def extract_embedding(model, feature_layer, image_raw):
    # Read the data in and normalize
    
    image_raw = normalizePercentile(image_raw)

    # Get the patch size from the input layer of the model
    feature_extractor = Model(inputs=model.input, outputs=feature_layer)
    patch_size = model.input_shape[1:3]

    # Pad the image with zeros if any of its dimensions is smaller than the patch size
    if image_raw.shape[0] < patch_size[0] or image_raw.shape[1] < patch_size[1]:
        Image = np.zeros((max(image_raw.shape[0], patch_size[0]), max(image_raw.shape[1], patch_size[1])))
        Image[0:image_raw.shape[0], 0:image_raw.shape[1]] = image_raw
    else:
        Image = image_raw

    # Calculate the number of patches in each dimension
    n_patch_in_width = ceil(Image.shape[0] / patch_size[0])
    n_patch_in_height = ceil(Image.shape[1] / patch_size[1])

    # Initialize a list to store feature vectors
    feature_vectors = []

    for x in range(n_patch_in_width):
        for y in range(n_patch_in_height):
            xi = patch_size[0] * x
            yi = patch_size[1] * y

            # If the patch exceeds the edge of the image, shift it back
            if xi + patch_size[0] >= Image.shape[0]:
                xi = Image.shape[0] - patch_size[0]

            if yi + patch_size[1] >= Image.shape[1]:
                yi = Image.shape[1] - patch_size[1]

            # Extract and reshape the patch
            patch = Image[xi:xi + patch_size[0], yi:yi + patch_size[1]]
            patch = np.reshape(patch, patch.shape + (1,))
            patch = np.reshape(patch, (1,) + patch.shape)

            # Get the feature vector from the patch
            feature_vector = feature_extractor.predict(patch, batch_size=1)

            # Store the feature vector with its position
            feature_vectors.append(feature_vector)

    return feature_vectors