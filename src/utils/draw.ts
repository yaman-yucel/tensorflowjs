import type { Detection } from './postprocess'

// Semi-transparent green overlay for the document mask
const MASK_R = 0
const MASK_G = 220
const MASK_B = 100
const MASK_A = 110  // out of 255 (~43% opacity)

// Bounding box stroke style
const BOX_COLOR = 'rgba(0, 220, 100, 0.9)'
const BOX_LINE_WIDTH = 2

/**
 * Clear the canvas and draw all detection masks + bounding boxes.
 */
export function drawDetections(
  canvas: HTMLCanvasElement,
  detections: Detection[],
): void {
  const ctx = canvas.getContext('2d')
  if (!ctx) return

  ctx.clearRect(0, 0, canvas.width, canvas.height)

  if (detections.length === 0) return

  for (const det of detections) {
    drawMask(ctx, det)
    drawBox(ctx, det)
  }
}

function drawMask(ctx: CanvasRenderingContext2D, det: Detection): void {
  const { mask, maskWidth, maskHeight } = det
  const imageData = ctx.createImageData(maskWidth, maskHeight)
  const data = imageData.data

  for (let i = 0; i < mask.length; i++) {
    if (mask[i] > 0.5) {
      const base = i * 4
      data[base]     = MASK_R
      data[base + 1] = MASK_G
      data[base + 2] = MASK_B
      data[base + 3] = MASK_A
    }
  }

  ctx.putImageData(imageData, 0, 0)
}

function drawBox(ctx: CanvasRenderingContext2D, det: Detection): void {
  const [y1, x1, y2, x2] = det.box
  ctx.strokeStyle = BOX_COLOR
  ctx.lineWidth = BOX_LINE_WIDTH
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1)

  // Score label
  const label = `doc ${(det.score * 100).toFixed(0)}%`
  ctx.font = 'bold 14px system-ui'
  const textWidth = ctx.measureText(label).width
  ctx.fillStyle = BOX_COLOR
  ctx.fillRect(x1, y1 - 20, textWidth + 8, 20)
  ctx.fillStyle = '#000'
  ctx.fillText(label, x1 + 4, y1 - 5)
}
