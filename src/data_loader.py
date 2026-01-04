from pathlib import Path
import tensorflow as tf

IMG_SIZE = 256
BATCH_SIZE = 16

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "mvtec"

def load_and_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels = 3)
    img = tf.image.resize (img, (256, 256))
    img = tf.cast(img, tf.float32)/255.0
    return img

def get_train_paths(category):
    train_dir = DATA_ROOT/ category / "train" / "good"
    return [str(p) for p in train_dir.glob("*.png")]

def get_test_paths(category):
    test_dir = DATA_ROOT / category / "test"
    paths = []
    labels = []                       # 0 = good 1 = anomaly

    for d in test_dir.iterdir():
        if not d.is_dir():
            continue

        imgs = [str(p) for p in d.glob("*.png")]
        paths.extend(imgs)

        if d.name == "good":
            labels.extend([0] * len(imgs))
        else:
            labels.extend([1] * len(imgs))

    return paths, labels


def build_train_dataset(image_paths):
    ds = tf.data.Dataset.from_tensor_slices(image_paths)
    ds = ds. shuffle(buffer_size=len(image_paths), seed = 21)
    ds = ds.map(load_and_preprocess,
                num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)

    return ds

def build_test_dataset(image_paths, labels):

    ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    ds = ds.map(
        lambda x, y: (load_and_preprocess(x),y), 
        num_parallel_calls= tf.data.AUTOTUNE
    )
    ds = ds.batch(BATCH_SIZE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds