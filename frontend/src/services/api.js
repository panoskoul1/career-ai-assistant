const BASE = '/api'

export async function uploadResume(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/upload/resume`, { method: 'POST', body: form })
  if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed')
  return res.json()
}

export async function uploadJob(file) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${BASE}/upload/job`, { method: 'POST', body: form })
  if (!res.ok) throw new Error((await res.json()).detail || 'Upload failed')
  return res.json()
}

export async function fetchJobs() {
  const res = await fetch(`${BASE}/jobs`)
  if (!res.ok) throw new Error('Failed to fetch jobs')
  return res.json()
}

export async function sendChat(message, jobId, sessionId) {
  const body = { message, session_id: sessionId }
  if (jobId) body.job_id = jobId
  const res = await fetch(`${BASE}/chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error((await res.json()).detail || 'Chat failed')
  return res.json()
}

export async function resetSession(sessionId) {
  await fetch(`${BASE}/session/${sessionId}`, { method: 'DELETE' })
}
