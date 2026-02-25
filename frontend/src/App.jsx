import React, { useState, useEffect, useCallback, useRef } from 'react'
import FileUpload from './components/FileUpload'
import JobSelector from './components/JobSelector'
import ChatWindow from './components/ChatWindow'
import { uploadResume, uploadJob, fetchJobs, sendChat } from './services/api'

function generateSessionId() {
  return 'session-' + Math.random().toString(36).slice(2) + Date.now().toString(36)
}

export default function App() {
  const [jobs, setJobs] = useState([])
  const [selectedJob, setSelectedJob] = useState(null)
  const [chatLoading, setChatLoading] = useState(false)

  // Stable session ID for the lifetime of this browser tab
  const sessionId = useRef(generateSessionId()).current

  const refreshJobs = useCallback(async () => {
    try {
      const data = await fetchJobs()
      setJobs(data)
    } catch {
      // silently ignore on initial load
    }
  }, [])

  useEffect(() => {
    refreshJobs()
  }, [refreshJobs])

  const handleUploadJob = async (file) => {
    const result = await uploadJob(file)
    await refreshJobs()
    return result
  }

  const handleChat = async (message) => {
    setChatLoading(true)
    try {
      return await sendChat(message, selectedJob, sessionId)
    } finally {
      setChatLoading(false)
    }
  }

  return (
    <div style={{ maxWidth: 720, margin: '0 auto', padding: '24px 16px', fontFamily: 'system-ui, sans-serif' }}>
      <h1 style={{ fontSize: 22, marginBottom: 4 }}>Career Intelligence Assistant</h1>
      <p style={{ color: '#666', marginTop: 0, marginBottom: 20, fontSize: 14 }}>
        Upload your resume and job descriptions to analyze fit, gaps, and interview prep.
      </p>

      <div style={{ display: 'flex', gap: 24, flexWrap: 'wrap', marginBottom: 16 }}>
        <div style={{ flex: 1, minWidth: 250 }}>
          <FileUpload label="Resume (PDF or text)" onUpload={uploadResume} />
        </div>
        <div style={{ flex: 1, minWidth: 250 }}>
          <FileUpload label="Job Description (PDF or text)" onUpload={handleUploadJob} />
        </div>
      </div>

      <JobSelector jobs={jobs} selectedJob={selectedJob} onSelect={setSelectedJob} />

      <ChatWindow onSend={handleChat} loading={chatLoading} />
    </div>
  )
}
