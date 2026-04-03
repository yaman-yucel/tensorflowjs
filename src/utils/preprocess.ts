import * as tf from '@tensorflow/tfjs'

/** Resize a video frame to [1, 640, 640, 3] float32 in [0, 1] range. */
export function preprocessFrame(video: HTMLVideoElement): tf.Tensor4D {
  return tf.tidy(() => {
    const img = tf.browser.fromPixels(video)              // [H, W, 3]
    const resized = tf.image.resizeBilinear(img, [640, 640]) // [640, 640, 3]
    const normalized = resized.div<tf.Tensor3D>(255)      // [0, 1]
    return normalized.expandDims<tf.Tensor4D>(0)          // [1, 640, 640, 3]
  })
}
