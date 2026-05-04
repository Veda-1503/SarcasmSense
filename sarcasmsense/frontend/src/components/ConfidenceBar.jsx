import React from 'react'

const sentimentColor = {
  positive: 'var(--positive)',
  neutral: 'var(--neutral)',
  negative: 'var(--negative)',
}

export default function ConfidenceBar({ value, sentiment }) {
  const color = sentimentColor[(sentiment || 'neutral').toLowerCase()] || 'var(--accent)'
  const pct = Math.round((value || 0) * 100)

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
      <div className="conf-bar-track" style={{ flex: 1 }}>
        <div
          className="conf-bar-fill"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span style={{
        fontFamily: 'var(--font-mono)',
        fontSize: 11,
        color: 'var(--text-muted)',
        flexShrink: 0,
      }}>{pct}%</span>
    </div>
  )
}
