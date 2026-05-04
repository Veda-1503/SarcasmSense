import React, { useState } from 'react'
import SentimentChip from '../components/SentimentChip.jsx'
import SarcasmBadge from '../components/SarcasmBadge.jsx'
import ConfidenceBar from '../components/ConfidenceBar.jsx'

const API = '/api'

export default function RealTimePage() {
  const [text, setText] = useState('')
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)

  async function handleAnalyze() {
    if (!text.trim()) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const res = await fetch(`${API}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text.trim() }),
      })
      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || 'Server error')
      }
      const data = await res.json()
      setResult(data)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  function handleKeyDown(e) {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) handleAnalyze()
  }

  return (
    <div>
      {/* Page title */}
      <div style={{ marginBottom: 28 }}>
        <h1 style={{
          fontFamily: 'var(--font-display)',
          fontSize: 22,
          fontWeight: 700,
          color: 'var(--text-primary)',
          letterSpacing: '-0.03em',
          marginBottom: 6,
        }}>Real-Time Analysis</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
          Enter a review to detect sarcasm and analyze sentiment across domain-specific models.
        </p>
      </div>

      {/* Input card */}
      <div className="card" style={{ padding: 20, marginBottom: 24 }}>
        <label style={{
          display: 'block',
          fontSize: 11,
          fontFamily: 'var(--font-mono)',
          color: 'var(--text-muted)',
          letterSpacing: '0.08em',
          marginBottom: 10,
          textTransform: 'uppercase',
        }}>Input Review</label>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Enter a review to analyze... (Ctrl+Enter to submit)"
          rows={4}
          style={{
            width: '100%',
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border-light)',
            borderRadius: 7,
            padding: '12px 14px',
            color: 'var(--text-primary)',
            fontFamily: 'var(--font-body)',
            fontSize: 14,
            resize: 'vertical',
            outline: 'none',
            transition: 'border-color 0.15s',
          }}
          onFocus={e => e.target.style.borderColor = 'var(--accent)'}
          onBlur={e => e.target.style.borderColor = 'var(--border-light)'}
        />
        <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: 12 }}>
          <button
            onClick={handleAnalyze}
            disabled={loading || !text.trim()}
            style={{
              padding: '9px 22px',
              background: loading || !text.trim() ? 'var(--bg-elevated)' : 'var(--accent)',
              color: loading || !text.trim() ? 'var(--text-muted)' : '#fff',
              border: 'none',
              borderRadius: 7,
              cursor: loading || !text.trim() ? 'not-allowed' : 'pointer',
              fontFamily: 'var(--font-display)',
              fontWeight: 600,
              fontSize: 13,
              letterSpacing: '-0.01em',
              transition: 'all 0.15s ease',
              display: 'flex', alignItems: 'center', gap: 8,
            }}
          >
            {loading ? (
              <>
                <span className="pulse-slow">●</span>
                Processing...
              </>
            ) : 'Analyze →'}
          </button>
        </div>
      </div>

      {/* Error */}
      {error && (
        <div style={{
          padding: '12px 16px',
          background: 'rgba(245,101,101,0.08)',
          border: '1px solid rgba(245,101,101,0.2)',
          borderRadius: 8,
          color: 'var(--negative)',
          fontSize: 13,
          marginBottom: 20,
        }}>
          ⚠ {error}
        </div>
      )}

      {/* Results */}
      {result && (
        <div className="fade-in">
          {/* Final Result Panel */}
          <FinalResultPanel result={result} />

          {/* Sentence Analysis */}
          <div style={{ marginTop: 24 }}>
            <h2 style={{
              fontFamily: 'var(--font-display)',
              fontSize: 14,
              fontWeight: 600,
              color: 'var(--text-secondary)',
              letterSpacing: '0.06em',
              textTransform: 'uppercase',
              marginBottom: 14,
            }}>Sentence Analysis</h2>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
              {result.sentence_results.map((sr, i) => (
                <SentenceCard key={i} sentence={sr} index={i + 1} />
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function FinalResultPanel({ result }) {
  const domainColors = {
    movies: '#a78bfa', healthcare: '#34d399', ecommerce: '#f59e0b',
    restaurants: '#fb923c', hotels: '#60a5fa', apps: '#5b8dee',
  }
  const domainColor = domainColors[result.domain] || 'var(--accent)'

  return (
    <div className="card" style={{ padding: 20 }}>
      <div style={{
        fontSize: 11,
        fontFamily: 'var(--font-mono)',
        color: 'var(--text-muted)',
        letterSpacing: '0.08em',
        textTransform: 'uppercase',
        marginBottom: 16,
      }}>Final Result</div>

      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))', gap: 16 }}>
        {/* Domain */}
        <Metric
          label="Detected Domain"
          value={<span style={{
            fontFamily: 'var(--font-display)',
            fontWeight: 700,
            fontSize: 18,
            color: domainColor,
            textTransform: 'capitalize',
          }}>{result.domain}</span>}
        />

        {/* Model */}
        <Metric
          label="Model Used"
          value={<span style={{
            fontFamily: 'var(--font-mono)',
            fontSize: 11,
            color: 'var(--text-secondary)',
          }}>{result.model_used}</span>}
        />

        {/* Overall Sarcasm */}
        <Metric
          label="Overall Sarcasm"
          value={
            <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <span className={`chip-${result.overall_sarcasm ? 'sarcasm' : 'no-sarcasm'}`}
                style={{ padding: '3px 10px', borderRadius: 99, fontSize: 12, fontWeight: 600 }}>
                {result.overall_sarcasm ? 'Yes' : 'No'}
              </span>
              {result.overall_sarcasm && (
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-muted)' }}>
                  {(result.overall_sarcasm_prob * 100).toFixed(0)}%
                </span>
              )}
            </div>
          }
        />

        {/* Final Sentiment */}
        <Metric
          label="Final Sentiment"
          value={
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <SentimentChip label={result.final_sentiment} large />
              <ConfidenceBar value={result.final_confidence} sentiment={result.final_sentiment} />
            </div>
          }
        />
      </div>
    </div>
  )
}

function Metric({ label, value }) {
  return (
    <div className="card-elevated" style={{ padding: '14px 16px' }}>
      <div style={{
        fontSize: 10,
        fontFamily: 'var(--font-mono)',
        color: 'var(--text-muted)',
        letterSpacing: '0.07em',
        textTransform: 'uppercase',
        marginBottom: 8,
      }}>{label}</div>
      {value}
    </div>
  )
}

function SentenceCard({ sentence: sr, index }) {
  return (
    <div className="card" style={{ padding: '16px 20px' }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', gap: 14 }}>
        {/* Index */}
        <div style={{
          flexShrink: 0,
          width: 26, height: 26,
          background: 'var(--bg-elevated)',
          border: '1px solid var(--border)',
          borderRadius: 6,
          display: 'flex', alignItems: 'center', justifyContent: 'center',
          fontFamily: 'var(--font-mono)',
          fontSize: 11,
          color: 'var(--text-muted)',
        }}>{index}</div>

        <div style={{ flex: 1, minWidth: 0 }}>
          {/* Sentence text */}
          <p style={{
            color: 'var(--text-primary)',
            fontSize: 14,
            marginBottom: 14,
            lineHeight: 1.55,
          }}>"{sr.sentence}"</p>

          {/* Badges row */}
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginBottom: 12 }}>
            <Badge label="Clause" value={sr.clause} mono />
            <SarcasmBadge result={sr.sarcasm} />
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Base:</span>
              <SentimentChip label={sr.sentiment.base_label} />
              <span style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
                {(sr.sentiment.base_confidence * 100).toFixed(0)}%
              </span>
            </div>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Final:</span>
              <SentimentChip label={sr.sentiment.final_label} />
            </div>
          </div>

          {/* Explanation */}
          <div style={{
            padding: '7px 12px',
            background: 'var(--bg-elevated)',
            borderRadius: 5,
            borderLeft: '2px solid var(--border-light)',
          }}>
            <span style={{
              fontSize: 11,
              fontFamily: 'var(--font-mono)',
              color: 'var(--text-muted)',
              letterSpacing: '0.01em',
            }}>{sr.explanation}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

function Badge({ label, value, mono }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
      <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{label}:</span>
      <span style={{
        fontFamily: mono ? 'var(--font-mono)' : 'var(--font-body)',
        fontSize: 12,
        color: 'var(--text-secondary)',
        background: 'var(--bg-elevated)',
        border: '1px solid var(--border)',
        padding: '2px 8px',
        borderRadius: 4,
      }}>{value}</span>
    </div>
  )
}
