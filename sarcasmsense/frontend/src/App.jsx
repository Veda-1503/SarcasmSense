import React, { useState } from 'react'
import RealTimePage from './pages/RealTimePage.jsx'
import EvaluationPage from './pages/EvaluationPage.jsx'

export default function App() {
  const [tab, setTab] = useState('realtime')

  return (
    <div style={{ minHeight: '100vh', background: 'var(--bg-base)' }}>
      {/* Header */}
      <header style={{
        borderBottom: '1px solid var(--border)',
        background: 'var(--bg-surface)',
        position: 'sticky',
        top: 0,
        zIndex: 50,
      }}>
        <div style={{ maxWidth: 1100, margin: '0 auto', padding: '0 24px' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 32, height: 56 }}>
            {/* Logo */}
            <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexShrink: 0 }}>
              <div style={{
                width: 28, height: 28,
                background: 'linear-gradient(135deg, var(--accent), #8b5cf6)',
                borderRadius: 7,
                display: 'flex', alignItems: 'center', justifyContent: 'center',
                fontSize: 14,
              }}>⚡</div>
              <span style={{
                fontFamily: 'var(--font-display)',
                fontWeight: 700,
                fontSize: 15,
                color: 'var(--text-primary)',
                letterSpacing: '-0.02em',
              }}>SarcasmSense</span>
            </div>

            {/* Nav */}
            <nav style={{ display: 'flex', gap: 4 }}>
              {[
                { key: 'realtime', label: 'Real-Time Analysis' },
                { key: 'evaluation', label: 'Evaluation' },
              ].map(({ key, label }) => (
                <button
                  key={key}
                  onClick={() => setTab(key)}
                  style={{
                    padding: '6px 14px',
                    borderRadius: 6,
                    border: 'none',
                    cursor: 'pointer',
                    fontFamily: 'var(--font-body)',
                    fontSize: 13,
                    fontWeight: tab === key ? 500 : 400,
                    background: tab === key ? 'var(--accent-dim)' : 'transparent',
                    color: tab === key ? 'var(--accent)' : 'var(--text-secondary)',
                    transition: 'all 0.15s ease',
                  }}
                >
                  {label}
                </button>
              ))}
            </nav>

            {/* Right label */}
            <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 8 }}>
              <span style={{
                fontFamily: 'var(--font-mono)',
                fontSize: 11,
                color: 'var(--text-muted)',
                letterSpacing: '0.05em',
              }}>DOMAIN-SPECIFIC · TRANSFORMER-BASED</span>
            </div>
          </div>
        </div>
      </header>

      {/* Page content */}
      <main style={{ maxWidth: 1100, margin: '0 auto', padding: '32px 24px' }}>
        {tab === 'realtime' ? <RealTimePage /> : <EvaluationPage />}
      </main>
    </div>
  )
}
