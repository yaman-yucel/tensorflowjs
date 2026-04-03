interface Props {
  message: string
}

export function LoadingScreen({ message }: Props) {
  return (
    <div className="fixed inset-0 flex flex-col items-center justify-center bg-black">
      {/* Spinner */}
      <div className="mb-6 h-14 w-14 animate-spin rounded-full border-4 border-white/20 border-t-green-400" />
      <p className="text-center text-base font-medium text-white/80">{message}</p>
    </div>
  )
}
