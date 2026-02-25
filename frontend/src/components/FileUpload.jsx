import React, { useRef, useState } from 'react'

export default function FileUpload({ label, onUpload, accept = '.pdf,.txt' }) {
  const inputRef = useRef(null)
  const [status, setStatus] = useState(null)
  const [loading, setLoading] = useState(false)

  const handleChange = async (e) => {
    const file = e.target.files[0]
    if (!file) return
    setLoading(true)
    setStatus(null)
    try {
      const result = await onUpload(file)
      setStatus({ ok: true, msg: `Uploaded: ${result.filename || file.name} (${result.chunks} chunks)` })
    } catch (err) {
      setStatus({ ok: false, msg: err.message })
    } finally {
      setLoading(false)
      if (inputRef.current) inputRef.current.value = ''
    }
  }

  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ fontWeight: 600 }}>{label}</label>
      <br />
      <input ref={inputRef} type="file" accept={accept} onChange={handleChange} disabled={loading} />
      {loading && <span style={{ marginLeft: 8, color: '#888' }}>Uploading...</span>}
      {status && (
        <div style={{ color: status.ok ? '#2a7' : '#c33', fontSize: 13, marginTop: 4 }}>
          {status.msg}
        </div>
      )}
    </div>
  )
}
