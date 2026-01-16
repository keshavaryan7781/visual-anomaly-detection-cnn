import tensorflow as tf
import numpy as np  
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from sklearn.metrics import pairwise_distances

def build_backbone():
    resnet = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=(256, 256, 3)
    )
    resnet.trainable = False

    layer_names = [
        "conv2_block3_out",
        "conv3_block4_out",
        "conv4_block6_out"
    ]

    outputs = [resnet.get_layer(name).output for name in layer_names]

    backbone = Model(
        inputs=resnet.input,
        outputs=outputs,
        name="patchcore_backbone"
    )
    return backbone

def align_features(feature_maps, target_size=(64, 64)):
    aligned = []
    for f in feature_maps:
        resized = tf.image.resize(f, target_size, method="bilinear")
        aligned.append(resized)
    return aligned

def concatenate_features(aligned_features):
    return tf.concat(aligned_features, axis=-1)

def extract_patches(feature_maps):
    b, h, w, c = feature_maps.shape
    return tf.reshape(feature_maps, (-1, c))

def build_memory_bank(patches, sampling_ratio=0.1):
    n_total = patches.shape[0]
    n_sample = int(n_total * sampling_ratio)
    idx = np.random.choice(n_total, n_sample, replace=False)
    return patches[idx]

def compute_patch_score(test_patches, memory_bank, batch_size=512):
    if isinstance(memory_bank, np.ndarray):
        memory_bank = tf.convert_to_tensor(memory_bank, dtype=tf.float32)

    n_patches = test_patches.shape[0]
    min_distances = []

    for i in range(0, n_patches, batch_size):
        chunk = test_patches[i : i + batch_size]

        sims = tf.matmul(chunk, memory_bank, transpose_b=True)

        dists = 2 - 2 * sims
        
        min_dist = tf.reduce_min(dists, axis=1)
        min_distances.append(min_dist)

    return tf.concat(min_distances, axis=0).numpy()

def image_anomaly_score(patch_scores):
    return patch_scores.max()

def build_anomaly_map(patch_scores, h=64, w=64):
    return patch_scores.reshape(h, w)