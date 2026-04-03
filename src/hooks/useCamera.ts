import { useEffect, useRef, useState } from 'react'

export type CameraState =
  | { status: 'requesting' }
  | { status: 'ready' }
  | { status: 'error'; message: string }

export function useCamera(
  videoRef: React.RefObject<HTMLVideoElement | null>,
  deviceId?: string,
): CameraState {
  const [state, setState] = useState<CameraState>({ status: 'requesting' })
  const streamRef = useRef<MediaStream | null>(null)

  useEffect(() => {
    let cancelled = false

    async function startCamera() {
      const videoEl = videoRef.current

      try {
        const constraints: MediaStreamConstraints = {
          video: deviceId
            ? { deviceId: { exact: deviceId } }
            : {
                facingMode: { ideal: 'environment' },
                width: { ideal: 1280 },
                height: { ideal: 720 },
              },
          audio: false,
        }

        const stream = await navigator.mediaDevices.getUserMedia(constraints)

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
        streamRef.current?.getTracks().forEach((t) => t.stop())
        streamRef.current = null

        if (!cancelled) {
          setState({
            status: 'error',
            message: err instanceof Error ? err.message : 'Camera access denied',
          })
        }
      }
    }

    setState({ status: 'requesting' })
    void startCamera()

    return () => {
      cancelled = true
      streamRef.current?.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
  }, [videoRef, deviceId])

  return state
}
