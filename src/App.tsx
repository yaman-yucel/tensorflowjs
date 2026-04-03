import { useModel } from './hooks/useModel'
import { CameraView } from './components/CameraView'
import { LoadingScreen } from './components/LoadingScreen'
import { ErrorScreen } from './components/ErrorScreen'

const MODEL_URL = '/model/model.json'

export default function App() {
  const modelState = useModel(MODEL_URL)

  if (modelState.status === 'loading') {
    return <LoadingScreen message="Loading segmentation model…" />
  }

  if (modelState.status === 'error') {
    return (
      <ErrorScreen
        title="Model failed to load"
        message={`${modelState.message} — make sure model.json and weight files are in /public/model/`}
      />
    )
  }

  return <CameraView model={modelState.model} />
}
