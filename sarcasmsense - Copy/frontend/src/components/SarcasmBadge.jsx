import React from 'react'

const levelColors = {
  Low: { color: '#f0b429', bg: 'rgba(240,180,41,0.1)', border: 'rgba(240,180,41,0.25)' },
  Medium: { color: '#fb923c', bg: 'rgba(251,146,60,0.1)', border: 'rgba(251,146,60,0.25)' },
  High: { color: '#f56565', bg: 'rgba(245,101,101,0.1)', border: 'rgba(245,101,101,0.25)' },
}

export default function SarcasmBadge({ result }) {
  if (!result) return null
  const isSarcasm = result.is_sarcasm
  const level = result.level || 'Low'
  const lc = levelColors[level] || levelColors.Low

  if (!isSarcasm) {
    return (
      <span className="chip-no-sarcasm" style={{
        padding: '2px 9px', borderRadius: 99, fontSize: 11, fontWeight: 600,
        fontFamily: 'var(--font-display)',
      }}>
        No Sarcasm
      </span>
    )
  }

  return (
    <span style={{
      display: 'inline-flex', alignItems: 'center', gap: 5,
      padding: '2px 9px', borderRadius: 99, fontSize: 11, fontWeight: 600,
      fontFamily: 'var(--font-display)',
      background: lc.bg, color: lc.color, border: `1px solid ${lc.border}`,
    }}>
      ⚡ Sarcasm · {level}
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, opacity: 0.8 }}>
        {(result.confidence * 100).toFixed(0)}%
      </span>
    </span>
  )
}
