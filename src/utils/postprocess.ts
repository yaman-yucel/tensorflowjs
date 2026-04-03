import * as tf from '@tensorflow/tfjs'

export const CONF_THRESHOLD = 0.5
export const IOU_THRESHOLD = 0.45
export const MAX_DETECTIONS = 10

export interface Detection {
  /** Bounding box in pixel coords [y1, x1, y2, x2] */
  box: [number, number, number, number]
  score: number
  /** Float32Array of length outputW * outputH with values in [0, 1] */
  mask: Float32Array
  maskWidth: number
  maskHeight: number
}

/**
 * Decode Ultralytics YOLO-seg TF.js output.
 *
 * output0: [1, 8400, 37]  — 4 (xywh) + 1 (conf) + 32 (mask coeffs)
 * output1: [1, 32, 160, 160] — prototype masks
 *
 * Returns detections scaled to (videoWidth × videoHeight).
 */
export async function postprocess(
  output0: tf.Tensor,
  output1: tf.Tensor,
  videoWidth: number,
  videoHeight: number,
): Promise<Detection[]> {
  // --- shapes ---
  // output0: [1, 37, 8400] OR [1, 8400, 37] depending on export version
  // Ultralytics tfjs export typically transposes to [1, 37, 8400]
  // We normalise to [8400, 37] below.

  const [preds, protos] = tf.tidy(() => {
    let raw = output0.squeeze([0]) as tf.Tensor2D // remove batch dim

    // Handle both [37, 8400] and [8400, 37]
    if (raw.shape[0] === 37) {
      raw = raw.transpose() as tf.Tensor2D // → [8400, 37]
    }

    const proto = output1.squeeze([0]) as tf.Tensor3D // [32, 160, 160]

    return [raw, proto]
  })

  // Pull raw predictions to CPU once.
  // Explicitly copy — on the CPU backend .data() returns a view into TF.js
  // internal memory that gets recycled after dispose().
  const rawData = new Float32Array(await preds.data<'float32'>())
  const numPreds = preds.shape[0]  // 8400
  const numCols = preds.shape[1]   // 37

  preds.dispose()

  // Filter by confidence
  const boxes: number[][] = []
  const scores: number[] = []
  const coeffsList: number[][] = []

  for (let i = 0; i < numPreds; i++) {
    const base = i * numCols
    const conf = rawData[base + 4]
    if (conf < CONF_THRESHOLD) continue

    const cx = rawData[base + 0]
    const cy = rawData[base + 1]
    const w  = rawData[base + 2]
    const h  = rawData[base + 3]

    // Normalised [0,1] → pixel coords
    const x1 = (cx - w / 2) * videoWidth
    const y1 = (cy - h / 2) * videoHeight
    const x2 = (cx + w / 2) * videoWidth
    const y2 = (cy + h / 2) * videoHeight

    // tf NMS expects [y1, x1, y2, x2] normalised to [0, 1].
    // Clamp so partially off-screen detections don't produce negative coords.
    const clamp = (v: number) => Math.max(0, Math.min(1, v))
    boxes.push([
      clamp(y1 / videoHeight),
      clamp(x1 / videoWidth),
      clamp(y2 / videoHeight),
      clamp(x2 / videoWidth),
    ])
    scores.push(conf)

    const coeffs: number[] = []
    for (let k = 5; k < numCols; k++) {
      coeffs.push(rawData[base + k])
    }
    coeffsList.push(coeffs)
  }

  if (boxes.length === 0) {
    protos.dispose()
    return []
  }

  // NMS
  const boxesTensor = tf.tensor2d(boxes)
  const scoresTensor = tf.tensor1d(scores)
  const nmsIndices = await tf.image.nonMaxSuppressionAsync(
    boxesTensor,
    scoresTensor,
    MAX_DETECTIONS,
    IOU_THRESHOLD,
    CONF_THRESHOLD,
  )
  const kept = await nmsIndices.data<'int32'>()
  boxesTensor.dispose()
  scoresTensor.dispose()
  nmsIndices.dispose()

  // Decode masks for kept detections
  const PROTO_SIZE = 160

  // protos: [32, 160, 160] → reshape to [32, 160*160]
  const protosFlat = protos.reshape([32, PROTO_SIZE * PROTO_SIZE]) as tf.Tensor2D
  protos.dispose()

  const detections: Detection[] = []

  for (const idx of kept) {
    // NMS should only return valid indices, but guard defensively
    if (idx >= coeffsList.length) continue

    const coeffsTensor = tf.tensor1d(coeffsList[idx]).reshape([1, 32]) as tf.Tensor2D
    // [1, 32] × [32, 160*160] → [1, 160*160]
    const maskFlat = tf.matMul(coeffsTensor, protosFlat) // [1, 25600]
    coeffsTensor.dispose()

    const maskSigmoid = tf.sigmoid(maskFlat.reshape([PROTO_SIZE, PROTO_SIZE]))
    maskFlat.dispose()

    // Resize mask to video dimensions
    const maskResized = tf.image.resizeBilinear(
      maskSigmoid.expandDims(-1) as tf.Tensor3D,
      [videoHeight, videoWidth],
    ).squeeze([-1]) as tf.Tensor2D

    maskSigmoid.dispose()

    // Copy data BEFORE dispose — on the CPU backend .data() returns a view
    // into the tensor's internal buffer, which gets recycled after dispose().
    const maskView = await maskResized.data<'float32'>()
    const maskData = new Float32Array(maskView)
    maskResized.dispose()

    const b = boxes[idx]
    detections.push({
      box: [
        b[0] * videoHeight,
        b[1] * videoWidth,
        b[2] * videoHeight,
        b[3] * videoWidth,
      ],
      score: scores[idx],
      mask: maskData,
      maskWidth: videoWidth,
      maskHeight: videoHeight,
    })
  }

  protosFlat.dispose()

  return detections
}
