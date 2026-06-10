import { useState } from "react"
import ImageUploader from "./components/ImageUploader"
import ResultViewer from "./components/ResultViewer"
import "./App.css"

export default function App() {
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleUpload = async (file) => {
    setLoading(true)
    setResult(null)

    const formData = new FormData()
    formData.append("image", file)

    try {
      const res = await fetch("http://localhost:8000/process", {
        method: "POST",
        body: formData,
      })
      
      if (!res.ok) {
        const data = await res.json().catch(() => ({}))
        const message =
          data.detail ||
          (res.status === 500
            ? "Image processing failed. Please try again."
            : "Upload failed. Please try again.")
        throw new Error(
          typeof message === "string" ? message : message[0]?.msg ?? "Upload failed."
        )
      }
      
      const blob = await res.blob()
      const output_url = URL.createObjectURL(blob)
      setResult({ output_url })
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  return (
    <main className="app">
      <section className="app-center">
        <ImageUploader onUpload={handleUpload} disabled={loading} />

        {loading && (
          <div className="status status--loading">
            <span className="spinner" />
            Processing...
          </div>
        )}

        {result && <ResultViewer result={result} />}
      </section>
    </main>
  )
}