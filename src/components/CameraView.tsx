import { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { useCamera } from '../hooks/useCamera'
import { preprocessFrame } from '../utils/preprocess'
import { postprocess } from '../utils/postprocess'
import { drawDetections } from '../utils/draw'
import { LoadingScreen } from './LoadingScreen'
import { ErrorScreen } from './ErrorScreen'

const INFERENCE_INTERVAL_MS = 200 // ~5 FPS

interface Props {
  model: tf.GraphModel
}

export function CameraView({ model }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [fps, setFps] = useState<number>(0)

  const cameraState = useCamera(videoRef)

  // Sync canvas size to actual video dimensions once metadata is loaded
  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    function onMetadata() {
      if (!canvasRef.current || !video) return
      canvasRef.current.width = video.videoWidth
      canvasRef.current.height = video.videoHeight
    }

    video.addEventListener('loadedmetadata', onMetadata)
    return () => video.removeEventListener('loadedmetadata', onMetadata)
  }, [])

  // Inference loop
  useEffect(() => {
    if (cameraState.status !== 'ready') return

    let cancelled = false
    let inFlight = false
    let lastTime = performance.now()

    const intervalId = setInterval(() => {
      // Drop frame if a previous inference is still running
      if (inFlight || cancelled) return

      const video = videoRef.current
      const canvas = canvasRef.current
      if (!video || !canvas || video.readyState < 2 || video.videoWidth === 0) return

      inFlight = true

      void (async () => {
        const input = preprocessFrame(video)
        let output0: tf.Tensor | undefined
        let output1: tf.Tensor | undefined

        try {
          // model.execute can return Tensor | Tensor[] | NamedTensorMap
          const raw = model.execute(input)

          if (Array.isArray(raw)) {
            ;[output0, output1] = raw
          } else if (raw instanceof tf.Tensor) {
            // Unexpected single-tensor output — not a YOLO-seg model
            raw.dispose()
            return
          } else {
            const outObj = raw as Record<string, tf.Tensor>
            output0 = outObj['output0']
            output1 = outObj['output1']
          }

          const detections = await postprocess(
            output0,
            output1,
            canvas.width,
            canvas.height,
          )

          // Don't update DOM after unmount
          if (cancelled) return

          drawDetections(canvas, detections)

          const now = performance.now()
          setFps(Math.round(1000 / (now - lastTime)))
          lastTime = now
        } finally {
          input.dispose()
          output0?.dispose()
          output1?.dispose()
          inFlight = false
        }
      })()
    }, INFERENCE_INTERVAL_MS)

    return () => {
      cancelled = true
      clearInterval(intervalId)
    }
  }, [cameraState.status, model])

  if (cameraState.status === 'requesting') {
    return <LoadingScreen message="Requesting camera access…" />
  }

  if (cameraState.status === 'error') {
    return (
      <ErrorScreen
        title="Camera unavailable"
        message={cameraState.message}
      />
    )
  }

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black">
      {/* Live camera feed */}
      <video
        ref={videoRef}
        className="absolute inset-0 h-full w-full object-cover"
        playsInline
        muted
        autoPlay
      />

      {/* Segmentation overlay — same object-cover treatment keeps it aligned */}
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full object-cover"
      />

      {/* HUD — bottom-0 + env(safe-area-inset-bottom) for iPhone notch */}
      <div
        className="absolute bottom-0 left-0 right-0 flex items-center justify-between px-4 pt-2"
        style={{ paddingBottom: 'max(env(safe-area-inset-bottom), 1.5rem)' }}
      >
        <span className="rounded-full bg-black/40 px-3 py-1 text-xs font-mono text-green-400 backdrop-blur-sm">
          {fps} FPS
        </span>
        <span className="rounded-full bg-black/40 px-3 py-1 text-xs font-medium text-white/70 backdrop-blur-sm">
          Doc Scanner
        </span>
      </div>
    </div>
  )
}
