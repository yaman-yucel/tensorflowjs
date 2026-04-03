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
    let loadedModel: tf.GraphModel | null = null

    async function load() {
      try {
        await tf.ready()
        const model = await tf.loadGraphModel(modelUrl)
        if (!cancelled) {
          loadedModel = model
          setState({ status: 'ready', model })
        } else {
          // Loaded after the effect was already cleaned up — dispose immediately
          model.dispose()
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
      loadedModel?.dispose()
    }
  }, [modelUrl])

  return state
}
