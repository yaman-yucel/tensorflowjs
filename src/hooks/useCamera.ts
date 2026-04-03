import { useEffect, useRef, useState } from 'react'

export type CameraState =
  | { status: 'requesting' }
  | { status: 'ready' }
  | { status: 'error'; message: string }

export function useCamera(videoRef: React.RefObject<HTMLVideoElement | null>): CameraState {
  const [state, setState] = useState<CameraState>({ status: 'requesting' })
  const streamRef = useRef<MediaStream | null>(null)

  useEffect(() => {
    let cancelled = false

    async function startCamera() {
      // Snapshot the element now — after the getUserMedia await the ref could
      // theoretically point to a different element (concurrent rendering).
      const videoEl = videoRef.current

      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            facingMode: { ideal: 'environment' },
            width: { ideal: 1280 },
            height: { ideal: 720 },
          },
          audio: false,
        })

        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop())
          return
        }

        streamRef.current = stream

        if (videoEl) {
          videoEl.srcObject = stream
          await videoEl.play()
        }

        setState({ status: 'ready' })
      } catch (err) {
        if (!cancelled) {
          setState({
            status: 'error',
            message:
              err instanceof Error ? err.message : 'Camera access denied',
          })
        }
      }
    }

    void startCamera()

    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
  }, [videoRef])

  return state
}
