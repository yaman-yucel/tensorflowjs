import * as tf from '@tensorflow/tfjs'

export const CONF_THRESHOLD = 0.5
export const MAX_DETECTIONS = 10

export interface Detection {
  /** Bounding box in pixel coords [y1, x1, y2, x2] */
  box: [number, number, number, number]
  score: number
  /** Float32Array of length maskWidth * maskHeight with values in [0, 1] */
  mask: Float32Array
  maskWidth: number
  maskHeight: number
}

/**
 * Decode end-to-end Ultralytics YOLO-seg TF.js output.
 *
 * The model has NMS baked in and emits two tensors:
 *   Identity:0   — [1, max_det, N] where N = 4 (xyxy, normalised 0-1)
 *                                             + 1 (score)
 *                                             + 1 (class id, cast to float) — optional
 *                                             + 32 (mask coefficients)
 *   Identity_1:0 — [1, 32, 160, 160] prototype masks
 *
 * Returns detections scaled to (videoWidth × videoHeight).
 */
export interface PostprocessDebug {
  numDets: number
  numCols: number
  maskStart: number
  protoLayout: 'channels-first' | 'channels-last'
  topScores: number[]
  firstBox: [number, number, number, number] | null
}

export async function postprocess(
  output0: tf.Tensor,
  output1: tf.Tensor,
  videoWidth: number,
  videoHeight: number,
  onDebug?: (d: PostprocessDebug) => void,
): Promise<Detection[]> {
  const predsRaw = output0.squeeze([0]) as tf.Tensor2D  // [max_det, N]
  const rawData = new Float32Array(await predsRaw.data<'float32'>())
  const numDets = predsRaw.shape[0]
  const numCols = predsRaw.shape[1]
  predsRaw.dispose()

  // Determine layout based on column count:
  //   37 = 4 (xyxy) + 1 (score) + 32 (mask coeffs)          — single-class, no class col
  //   38 = 4 (xyxy) + 1 (score) + 1 (class) + 32 (mask coeffs) — with class col
  const scoreCol = 4
  const maskStart = numCols >= 38 ? 6 : 5   // skip class col when present

  // output1 may be channels-first [1, 32, 160, 160] or channels-last [1, 160, 160, 32]
  // depending on the export. Detect layout from shape and normalise to [32, H*W].
  const protosRaw = output1.squeeze([0]) as tf.Tensor3D
  const PROTO_SIZE = 160
  let protosFlat: tf.Tensor2D
  let protoLayout: 'channels-first' | 'channels-last'
  if (protosRaw.shape[0] === 32) {
    protoLayout = 'channels-first'
    protosFlat = protosRaw.reshape([32, PROTO_SIZE * PROTO_SIZE]) as tf.Tensor2D
    protosRaw.dispose()
  } else {
    protoLayout = 'channels-last'
    const protosT = protosRaw.transpose([2, 0, 1]) as tf.Tensor3D
    protosRaw.dispose()
    protosFlat = protosT.reshape([32, PROTO_SIZE * PROTO_SIZE]) as tf.Tensor2D
    protosT.dispose()
  }

  // Collect info for debug callback
  const topScores: number[] = []
  for (let i = 0; i < Math.min(numDets, 5); i++) {
    topScores.push(rawData[i * numCols + scoreCol])
  }
  const firstBox: [number, number, number, number] | null = numDets > 0
    ? [rawData[0], rawData[1], rawData[2], rawData[3]]
    : null
  onDebug?.({ numDets, numCols, maskStart, protoLayout, topScores, firstBox })

  const detections: Detection[] = []

  try {
  for (let i = 0; i < numDets && detections.length < MAX_DETECTIONS; i++) {
    const base = i * numCols
    const score = rawData[base + scoreCol]
    if (score < CONF_THRESHOLD) continue

    // xyxy boxes are in the 640×640 input pixel space — normalise then scale
    const INPUT_SIZE = 640
    const x1 = (rawData[base + 0] / INPUT_SIZE) * videoWidth
    const y1 = (rawData[base + 1] / INPUT_SIZE) * videoHeight
    const x2 = (rawData[base + 2] / INPUT_SIZE) * videoWidth
    const y2 = (rawData[base + 3] / INPUT_SIZE) * videoHeight

    // Decode mask: coeffs × protos → sigmoid → resize
    // Each intermediate must be named and disposed — chained ops leak tensors.
    const coeffs = Array.from(rawData.subarray(base + maskStart, base + maskStart + 32))
    const coeffsTensor = tf.tensor1d(coeffs).reshape([1, 32]) as tf.Tensor2D

    const maskFlat = tf.matMul(coeffsTensor, protosFlat)  // [1, 25600]
    coeffsTensor.dispose()

    const maskFlatSq = maskFlat.reshape([PROTO_SIZE, PROTO_SIZE]) as tf.Tensor2D
    maskFlat.dispose()

    const maskSigmoid = tf.sigmoid(maskFlatSq) as tf.Tensor2D
    maskFlatSq.dispose()

    const maskExpanded = maskSigmoid.expandDims(-1) as tf.Tensor3D
    maskSigmoid.dispose()

    const maskResizedRaw = tf.image.resizeBilinear(maskExpanded, [videoHeight, videoWidth])
    maskExpanded.dispose()

    const maskResized = maskResizedRaw.squeeze([-1]) as tf.Tensor2D
    maskResizedRaw.dispose()

    const maskView = await maskResized.data<'float32'>()
    const maskData = new Float32Array(maskView)
    maskResized.dispose()

    detections.push({
      box: [y1, x1, y2, x2],
      score,
      mask: maskData,
      maskWidth: videoWidth,
      maskHeight: videoHeight,
    })
  }

  } finally {
    protosFlat.dispose()
  }

  return detections
}
