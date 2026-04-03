import { useEffect, useRef, useState } from 'react'
import * as tf from '@tensorflow/tfjs'
import { useCamera } from '../hooks/useCamera'
import { preprocessFrame } from '../utils/preprocess'
import { postprocess, type PostprocessDebug } from '../utils/postprocess'
import { drawDetections } from '../utils/draw'
import { LoadingScreen } from './LoadingScreen'
import { ErrorScreen } from './ErrorScreen'

const INFERENCE_INTERVAL_MS = 200 // ~5 FPS

interface Props {
  model: tf.GraphModel
}

interface DebugInfo {
  // Video
  videoReadyState: number
  videoWidth: number
  videoHeight: number
  videoPaused: boolean
  videoCurrentTime: number
  videoSrcSet: boolean
  // Canvas
  canvasWidth: number
  canvasHeight: number
  // Inference
  loopSkipReason: string
  lastInferenceError: string
  output0Shape: number[]
  output1Shape: number[]
  detectionCount: number
  // Model internals
  ppDebug: PostprocessDebug | null
  // TF.js
  backend: string
  numTensors: number
  numBytes: number
}

function makeDebugSnapshot(
  video: HTMLVideoElement | null,
  canvas: HTMLCanvasElement | null,
  skipReason: string,
  lastError: string,
  output0Shape: number[],
  output1Shape: number[],
  detectionCount: number,
  ppDebug: PostprocessDebug | null,
): DebugInfo {
  return {
    videoReadyState: video?.readyState ?? -1,
    videoWidth: video?.videoWidth ?? 0,
    videoHeight: video?.videoHeight ?? 0,
    videoPaused: video?.paused ?? true,
    videoCurrentTime: video?.currentTime ?? 0,
    videoSrcSet: !!(video?.srcObject),
    canvasWidth: canvas?.width ?? 0,
    canvasHeight: canvas?.height ?? 0,
    loopSkipReason: skipReason,
    lastInferenceError: lastError,
    output0Shape,
    output1Shape,
    detectionCount,
    ppDebug,
    backend: tf.getBackend() ?? 'unknown',
    numTensors: tf.memory().numTensors,
    numBytes: tf.memory().numBytes,
  }
}

export function CameraView({ model }: Props) {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [fps, setFps] = useState<number>(0)
  const [debugVisible, setDebugVisible] = useState(true)
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null)

  // Camera selection
  const [cameras, setCameras] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState<string | undefined>(undefined)
  const [pickerVisible, setPickerVisible] = useState(false)

  // Enumerate cameras once on mount (needs at least one getUserMedia call first
  // so the browser reveals device labels — useCamera does that for us)
  useEffect(() => {
    async function enumerate() {
      try {
        const devices = await navigator.mediaDevices.enumerateDevices()
        const videoDevices = devices.filter((d) => d.kind === 'videoinput')
        setCameras(videoDevices)
      } catch {
        // enumeration is best-effort
      }
    }

    // Run once now, then again after the first stream is granted (labels appear)
    void enumerate()
    navigator.mediaDevices.addEventListener('devicechange', enumerate)
    return () => navigator.mediaDevices.removeEventListener('devicechange', enumerate)
  }, [])

  const cameraState = useCamera(videoRef, selectedDeviceId)

  // Re-enumerate after first stream is granted — labels are hidden until then
  useEffect(() => {
    if (cameraState.status !== 'ready') return
    navigator.mediaDevices.enumerateDevices().then((devices) => {
      setCameras(devices.filter((d) => d.kind === 'videoinput'))
    }).catch(() => {})
  }, [cameraState.status])

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
    let lastError = ''
    let lastOutput0Shape: number[] = []
    let lastOutput1Shape: number[] = []
    let lastDetectionCount = 0
    let lastPpDebug: PostprocessDebug | null = null

    const intervalId = setInterval(() => {
      // Drop frame if a previous inference is still running
      if (inFlight || cancelled) return

      const video = videoRef.current
      const canvas = canvasRef.current

      // Diagnose why we might skip, then update debug if visible
      let skipReason = ''
      if (!video) skipReason = 'videoRef is null'
      else if (!canvas) skipReason = 'canvasRef is null'
      else if (video.readyState < 2) skipReason = `video.readyState=${video.readyState} (need ≥2)`
      else if (video.videoWidth === 0) skipReason = 'video.videoWidth=0 (metadata not loaded)'
      else if (canvas.width === 0) skipReason = 'canvas.width=0 (loadedmetadata not fired)'

      if (skipReason) {
        setDebugInfo(makeDebugSnapshot(video, canvas, skipReason, lastError, lastOutput0Shape, lastOutput1Shape, lastDetectionCount, lastPpDebug))
        return
      }

      inFlight = true

      void (async () => {
        let input: tf.Tensor | undefined
        let output0: tf.Tensor | undefined
        let output1: tf.Tensor | undefined

        try {
          // preprocessFrame inside try so any synchronous throw is caught
          input = preprocessFrame(video!)
          // End-to-end model exposes three signature outputs; request only the two
          // we need so the third (TopKV2 scores) is never allocated on the JS side.
          const raw = model.execute(input!, ['Identity:0', 'Identity_1:0']) as tf.Tensor[]
          ;[output0, output1] = raw

          lastOutput0Shape = output0.shape
          lastOutput1Shape = output1.shape

          const detections = await postprocess(
            output0,
            output1,
            canvas!.width,
            canvas!.height,
            (d) => { lastPpDebug = d },
          )

          // Don't update DOM after unmount
          if (cancelled) return

          lastDetectionCount = detections.length
          lastError = ''
          drawDetections(canvas!, detections)

          const now = performance.now()
          setFps(Math.round(1000 / (now - lastTime)))
          lastTime = now

          setDebugInfo(makeDebugSnapshot(video, canvas, '', lastError, lastOutput0Shape, lastOutput1Shape, lastDetectionCount, lastPpDebug))
        } catch (err) {
          lastError = err instanceof Error ? err.message : String(err)
          setDebugInfo(makeDebugSnapshot(video, canvas, '', lastError, lastOutput0Shape, lastOutput1Shape, lastDetectionCount, lastPpDebug))
        } finally {
          input?.dispose()
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

  return (
    <div className="relative h-screen w-screen overflow-hidden bg-black">
      {/* Live camera feed — always mounted so videoRef is set before useCamera runs */}
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

      {/* Loading / error overlays on top of the (invisible) video */}
      {cameraState.status === 'requesting' && (
        <div className="absolute inset-0 flex items-center justify-center bg-black">
          <LoadingScreen message="Requesting camera access…" />
        </div>
      )}
      {cameraState.status === 'error' && (
        <div className="absolute inset-0 flex items-center justify-center bg-black">
          <ErrorScreen title="Camera unavailable" message={cameraState.message} />
        </div>
      )}

      {/* Debug panel */}
      {debugVisible && (
        <div className="absolute left-0 top-0 bottom-0 w-72 overflow-y-auto p-3 font-mono text-xs text-white" style={{ background: 'rgba(0,0,0,0.85)' }}>
          <p className="mb-2 font-bold text-sm" style={{ color: '#facc15' }}>DEBUG (tap FPS to hide)</p>

          <Section label="Camera state">
            <Row label="status" value={cameraState.status} warn={cameraState.status !== 'ready'} />
            {'message' in cameraState && <Row label="error" value={cameraState.message} warn />}
          </Section>

          {debugInfo ? (
            <>
              <Section label="Video">
                <Row label="srcObject set" value={String(debugInfo.videoSrcSet)} warn={!debugInfo.videoSrcSet} />
                <Row label="readyState" value={`${debugInfo.videoReadyState} (need ≥2)`} warn={debugInfo.videoReadyState < 2} />
                <Row label="dimensions" value={`${debugInfo.videoWidth} × ${debugInfo.videoHeight}`} warn={debugInfo.videoWidth === 0} />
                <Row label="paused" value={String(debugInfo.videoPaused)} warn={debugInfo.videoPaused} />
                <Row label="currentTime" value={debugInfo.videoCurrentTime.toFixed(2) + 's'} />
              </Section>

              <Section label="Canvas">
                <Row label="dimensions" value={`${debugInfo.canvasWidth} × ${debugInfo.canvasHeight}`} warn={debugInfo.canvasWidth === 0} />
              </Section>

              <Section label="Inference loop">
                <Row label="skip reason" value={debugInfo.loopSkipReason || 'none'} warn={!!debugInfo.loopSkipReason} />
                <Row label="last error" value={debugInfo.lastInferenceError || 'none'} warn={!!debugInfo.lastInferenceError} />
                <Row label="output0 shape" value={debugInfo.output0Shape.length ? `[${debugInfo.output0Shape}]` : 'n/a'} />
                <Row label="output1 shape" value={debugInfo.output1Shape.length ? `[${debugInfo.output1Shape}]` : 'n/a'} />
                <Row label="detections" value={String(debugInfo.detectionCount)} />
                <Row label="FPS" value={String(fps)} />
              </Section>

              {debugInfo.ppDebug && (
                <Section label="Model output">
                  <Row label="numDets" value={String(debugInfo.ppDebug.numDets)} />
                  <Row label="numCols" value={String(debugInfo.ppDebug.numCols)} />
                  <Row label="maskStart" value={String(debugInfo.ppDebug.maskStart)} />
                  <Row label="protoLayout" value={debugInfo.ppDebug.protoLayout} />
                  <Row label="top scores" value={debugInfo.ppDebug.topScores.map(s => s.toFixed(2)).join(', ')} warn={debugInfo.ppDebug.topScores[0] > 0.5} />
                  <Row
                    label="first box"
                    value={debugInfo.ppDebug.firstBox
                      ? debugInfo.ppDebug.firstBox.map(v => v.toFixed(1)).join(', ')
                      : 'n/a'}
                  />
                </Section>
              )}

              <Section label="TF.js">
                <Row label="backend" value={debugInfo.backend} />
                <Row label="tensors" value={String(debugInfo.numTensors)} />
                <Row label="memory" value={`${(debugInfo.numBytes / 1024 / 1024).toFixed(1)} MB`} />
              </Section>
            </>
          ) : (
            <p style={{ color: '#94a3b8' }}>Waiting for first inference tick…</p>
          )}
        </div>
      )}

      {/* Camera picker — drops down below the HUD */}
      {pickerVisible && cameras.length > 0 && (
        <div className="absolute inset-x-4 top-14 rounded-xl overflow-hidden" style={{ background: 'rgba(0,0,0,0.85)' }}>
          {cameras.map((cam, i) => (
            <button
              key={cam.deviceId}
              onClick={() => { setSelectedDeviceId(cam.deviceId); setPickerVisible(false) }}
              className="w-full px-4 py-3 text-left text-sm text-white flex items-center gap-3"
              style={{ borderTop: i > 0 ? '1px solid rgba(255,255,255,0.1)' : undefined }}
            >
              <span style={{ color: cam.deviceId === (selectedDeviceId ?? '') ? '#4ade80' : 'transparent' }}>✓</span>
              <span>{cam.label || `Camera ${i + 1}`}</span>
            </button>
          ))}
        </div>
      )}

      {/* HUD — top with safe area inset */}
      <div
        className="absolute top-0 left-0 right-0 flex items-center justify-between px-4 pb-2"
        style={{ paddingTop: 'max(env(safe-area-inset-top), 0.75rem)' }}
      >
        <button
          onClick={() => setDebugVisible(v => !v)}
          className="rounded-full bg-black/40 px-3 py-1 text-xs font-mono text-green-400 backdrop-blur-sm"
        >
          {fps} FPS
        </button>
        <span className="rounded-full bg-black/40 px-3 py-1 text-xs font-medium text-white/70 backdrop-blur-sm">
          Doc Scanner
        </span>
        {cameras.length > 1 && (
          <button
            onClick={() => setPickerVisible(v => !v)}
            className="rounded-full bg-black/40 px-3 py-1 text-xs font-medium text-white/70 backdrop-blur-sm"
          >
            Camera {cameras.findIndex(c => c.deviceId === selectedDeviceId) + 1 || '↕'}
          </button>
        )}
      </div>
    </div>
  )
}

function Section({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="mb-3">
      <p className="mb-1 text-yellow-300/80 uppercase tracking-wide text-[10px]">{label}</p>
      <div className="space-y-0.5">{children}</div>
    </div>
  )
}

function Row({ label, value, warn }: { label: string; value: string; warn?: boolean }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-white/50 shrink-0">{label}</span>
      <span className={warn ? 'text-red-400 font-bold' : 'text-white/90'} style={{ wordBreak: 'break-all' }}>{value}</span>
    </div>
  )
}
