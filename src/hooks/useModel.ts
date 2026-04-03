import { useEffect, useState } from 'react'
import * as tf from '@tensorflow/tfjs'

export type ModelState =
  | { status: 'loading' }
  | { status: 'ready'; model: tf.GraphModel }
  | { status: 'error'; message: string }

export function useModel(modelUrl: string): ModelState {
  const [state, setState] = useState<ModelState>({ status: 'loading' })

  useEffect(() => {
    let cancelled = false

    async function load() {
      try {
        await tf.ready()
        const model = await tf.loadGraphModel(modelUrl)
        if (!cancelled) {
          setState({ status: 'ready', model })
        }
      } catch (err) {
        if (!cancelled) {
          setState({
            status: 'error',
            message: err instanceof Error ? err.message : 'Failed to load model',
          })
        }
      }
    }

    void load()

    return () => {
      cancelled = true
    }
  }, [modelUrl])

  return state
}
