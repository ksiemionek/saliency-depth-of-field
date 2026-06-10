import { useRef, useState } from "react"

export default function ImageUploader({ onUpload, disabled = false }) {
  const inputRef = useRef(null)
  const [dragOver, setDragOver] = useState(false)
  const [fileName, setFileName] = useState(null)

  const handleFile = (file) => {
    if (!file || !file.type.startsWith("image/")) return
    setFileName(file.name)
    onUpload(file)
  }

  const handleChange = (e) => {
    const file = e.target.files?.[0]
    if (file) handleFile(file)
  }

  const handleDrop = (e) => {
    e.preventDefault()
    setDragOver(false)
    if (disabled) return
    const file = e.dataTransfer.files?.[0]
    if (file) handleFile(file)
  }

  return (
    <div
      className={`upload-card${dragOver ? " upload-card--drag" : ""}${disabled ? " upload-card--disabled" : ""}`}
      onDragOver={(e) => {
        e.preventDefault()
        if (!disabled) setDragOver(true)
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={handleDrop}
    >
      <div className="upload-icon" aria-hidden="true">
        <svg width="40" height="40" viewBox="0 0 24 24" fill="none">
          <path
            d="M12 16V4m0 0l-4 4m4-4l4 4M4 20h16"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </svg>
      </div>

      <p className="upload-title">
        Choose a file or drag and drop it here
      </p>

      <input
        ref={inputRef}
        type="file"
        accept="image/jpeg,image/png,image/*"
        onChange={handleChange}
        disabled={disabled}
        hidden
      />

      <button
        type="button"
        className="upload-btn"
        disabled={disabled}
        onClick={() => inputRef.current?.click()}
      >
        Browse File
      </button>

      {fileName && <p className="upload-filename">{fileName}</p>}
    </div>
  )
}