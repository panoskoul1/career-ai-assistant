import React from 'react'

export default function JobSelector({ jobs, selectedJob, onSelect }) {
  if (jobs.length === 0) {
    return <div style={{ color: '#888', fontSize: 13 }}>No jobs uploaded yet.</div>
  }

  return (
    <div style={{ marginBottom: 12 }}>
      <label style={{ fontWeight: 600 }}>Compare against job:</label>
      <br />
      <select
        value={selectedJob || ''}
        onChange={(e) => onSelect(e.target.value || null)}
        style={{ padding: '6px 10px', fontSize: 14, marginTop: 4, minWidth: 200 }}
      >
        <option value="">All jobs (general)</option>
        {jobs.map((j) => (
          <option key={j.job_id} value={j.job_id}>
            {j.title} ({j.filename})
          </option>
        ))}
      </select>
    </div>
  )
}
