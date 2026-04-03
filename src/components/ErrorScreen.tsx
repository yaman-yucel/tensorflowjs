interface Props {
  title: string
  message: string
}

export function ErrorScreen({ title, message }: Props) {
  return (
    <div className="fixed inset-0 flex flex-col items-center justify-center gap-4 bg-black p-8">
      <div className="flex h-16 w-16 items-center justify-center rounded-full bg-red-500/20">
        <svg
          className="h-8 w-8 text-red-400"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M12 9v4m0 4h.01M10.29 3.86L1.82 18a2 2 0 001.71 3h16.94a2 2 0 001.71-3L13.71 3.86a2 2 0 00-3.42 0z"
          />
        </svg>
      </div>
      <h2 className="text-xl font-semibold text-white">{title}</h2>
      <p className="text-center text-sm text-white/60">{message}</p>
      <button
        onClick={() => window.location.reload()}
        className="mt-2 rounded-lg bg-white/10 px-6 py-2.5 text-sm font-medium text-white active:bg-white/20"
      >
        Try again
      </button>
    </div>
  )
}
