import React from 'react'

export default function SentimentChip({ label, large }) {
  const normalized = (label || 'neutral').toLowerCase()
  const cls = `chip-${normalized}`
  return (
    <span
      className={cls}
      style={{
        padding: large ? '4px 12px' : '2px 9px',
        borderRadius: 99,
        fontSize: large ? 13 : 11,
        fontWeight: 600,
        fontFamily: 'var(--font-display)',
        letterSpacing: '-0.01em',
        textTransform: 'capitalize',
        display: 'inline-block',
      }}
    >
      {normalized}
    </span>
  )
}
