import React, { useState } from 'react'

export default function ChatWindow({ onSend, loading }) {
  const [input, setInput] = useState('')
  const [messages, setMessages] = useState([])

  const handleSend = async () => {
    const text = input.trim()
    if (!text || loading) return
    setInput('')
    setMessages((prev) => [...prev, { role: 'user', content: text }])
    try {
      const res = await onSend(text)
      setMessages((prev) => [...prev, { role: 'assistant', content: res.answer }])
    } catch (err) {
      setMessages((prev) => [...prev, { role: 'error', content: err.message }])
    }
  }

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div style={{ border: '1px solid #ddd', borderRadius: 8, padding: 16, marginTop: 16 }}>
      <div
        style={{
          maxHeight: 400,
          overflowY: 'auto',
          marginBottom: 12,
          padding: 8,
          background: '#fafafa',
          borderRadius: 4,
          minHeight: 100,
        }}
      >
        {messages.length === 0 && (
          <div style={{ color: '#aaa', fontStyle: 'italic' }}>
            Upload a resume and job description, then ask questions here.
          </div>
        )}
        {messages.map((m, i) => (
          <div
            key={i}
            style={{
              marginBottom: 10,
              padding: '8px 12px',
              borderRadius: 6,
              background: m.role === 'user' ? '#e3f2fd' : m.role === 'error' ? '#fde3e3' : '#f1f8e9',
              whiteSpace: 'pre-wrap',
              wordBreak: 'break-word',
            }}
          >
            <strong>{m.role === 'user' ? 'You' : m.role === 'error' ? 'Error' : 'Assistant'}:</strong>
            <div style={{ marginTop: 4 }}>{m.content}</div>
          </div>
        ))}
        {loading && (
          <div style={{ color: '#888', fontStyle: 'italic' }}>Thinking...</div>
        )}
      </div>
      <div style={{ display: 'flex', gap: 8 }}>
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about skill gaps, strengths, interview prep..."
          style={{
            flex: 1,
            padding: '8px 12px',
            fontSize: 14,
            borderRadius: 4,
            border: '1px solid #ccc',
            resize: 'vertical',
            minHeight: 40,
            maxHeight: 120,
            fontFamily: 'inherit',
          }}
          disabled={loading}
        />
        <button
          onClick={handleSend}
          disabled={loading || !input.trim()}
          style={{
            padding: '8px 20px',
            fontSize: 14,
            borderRadius: 4,
            border: 'none',
            background: loading ? '#ccc' : '#1976d2',
            color: '#fff',
            cursor: loading ? 'default' : 'pointer',
          }}
        >
          Send
        </button>
      </div>
    </div>
  )
}
