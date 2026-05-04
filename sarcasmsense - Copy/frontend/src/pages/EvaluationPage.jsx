import React, { useState, useRef } from 'react'

const API = '/api'

export default function EvaluationPage() {
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState(null)
  const [error, setError] = useState(null)
  const [dragOver, setDragOver] = useState(false)
  const [fileName, setFileName] = useState(null)
  const fileRef = useRef()

  async function handleFile(file) {
    if (!file) return
    setFileName(file.name)
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API}/evaluate`, { method: 'POST', body: form })
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

  function handleDrop(e) {
    e.preventDefault()
    setDragOver(false)
    const file = e.dataTransfer.files[0]
    if (file && file.name.endsWith('.csv')) handleFile(file)
    else setError('Please upload a valid CSV file.')
  }

  return (
    <div>
      <div style={{ marginBottom: 28 }}>
        <h1 style={{
          fontFamily: 'var(--font-display)', fontSize: 22, fontWeight: 700,
          color: 'var(--text-primary)', letterSpacing: '-0.03em', marginBottom: 6,
        }}>Evaluation</h1>
        <p style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
          Upload a CSV with <code style={{ fontFamily: 'var(--font-mono)', background: 'var(--bg-elevated)', padding: '1px 5px', borderRadius: 3 }}>review, domain, final_sentiment</code> columns to evaluate system performance.
        </p>
      </div>

      {/* Upload */}
      <div
        className="card"
        style={{
          padding: 32, textAlign: 'center', cursor: 'pointer',
          borderStyle: 'dashed',
          borderColor: dragOver ? 'var(--accent)' : 'var(--border)',
          background: dragOver ? 'var(--accent-dim)' : 'var(--bg-card)',
          transition: 'all 0.15s ease',
          marginBottom: 24,
        }}
        onDragOver={e => { e.preventDefault(); setDragOver(true) }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileRef.current.click()}
      >
        <input
          ref={fileRef} type="file" accept=".csv" style={{ display: 'none' }}
          onChange={e => handleFile(e.target.files[0])}
        />
        <div style={{ fontSize: 28, marginBottom: 10 }}>📊</div>
        <div style={{ fontFamily: 'var(--font-display)', fontSize: 14, fontWeight: 600, color: 'var(--text-primary)', marginBottom: 4 }}>
          {fileName || 'Drop CSV file here or click to browse'}
        </div>
        <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
          Required columns: review, domain, final_sentiment
        </div>
      </div>

      {/* Loading */}
      {loading && (
        <div style={{
          padding: '20px', textAlign: 'center',
          background: 'var(--bg-card)', border: '1px solid var(--border)', borderRadius: 10,
          marginBottom: 20,
        }}>
          <div className="pulse-slow" style={{ fontSize: 13, color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}>
            Running evaluation pipeline — this may take several minutes...
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          padding: '12px 16px', background: 'rgba(245,101,101,0.08)',
          border: '1px solid rgba(245,101,101,0.2)', borderRadius: 8,
          color: 'var(--negative)', fontSize: 13, marginBottom: 20,
        }}>⚠ {error}</div>
      )}

      {/* Results */}
      {result && (
        <div className="fade-in">
          {/* Domain-wise table */}
          <Section title="Domain-Wise Performance">
            <DomainTable data={result.domain_results} />
          </Section>

          {/* Sarcasm Comparison */}
          <Section title="Sarcasm Impact Analysis" style={{ marginTop: 24 }}>
            <SarcasmComparison
              withSarcasm={result.overall_with_sarcasm}
              withoutSarcasm={result.overall_without_sarcasm}
              gain={result.performance_gain}
            />
          </Section>

          {/* Overall Metrics */}
          <Section title="Overall Performance" style={{ marginTop: 24 }}>
            <OverallMetrics data={result.overall_with_sarcasm} />
          </Section>

          {/* Confusion Matrix */}
          <Section title="Confusion Matrix" style={{ marginTop: 24 }}>
            <ConfMatrix data={result.confusion_matrix} />
          </Section>
        </div>
      )}
    </div>
  )
}

function Section({ title, children, style }) {
  return (
    <div className="card" style={{ padding: 20, ...style }}>
      <div style={{
        fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)',
        letterSpacing: '0.08em', textTransform: 'uppercase', marginBottom: 16,
      }}>{title}</div>
      {children}
    </div>
  )
}

function DomainTable({ data }) {
  const cols = [
    { key: 'domain', label: 'Domain' },
    { key: 'model', label: 'Model' },
    { key: 'sample_count', label: 'Samples' },
    { key: 'accuracy', label: 'Accuracy', pct: true },
    { key: 'macro_precision', label: 'Precision' },
    { key: 'macro_recall', label: 'Recall' },
    { key: 'macro_f1', label: 'Macro F1' },
    { key: 'weighted_f1', label: 'Weighted F1' },
  ]

  return (
    <div style={{ overflowX: 'auto' }}>
      <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr>
            {cols.map(c => (
              <th key={c.key} style={{
                textAlign: 'left', padding: '8px 12px',
                fontFamily: 'var(--font-mono)', fontSize: 10,
                color: 'var(--text-muted)', letterSpacing: '0.06em',
                textTransform: 'uppercase',
                borderBottom: '1px solid var(--border)',
                whiteSpace: 'nowrap',
              }}>{c.label}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.map((row, i) => (
            <tr key={i} style={{
              borderBottom: '1px solid var(--border)',
              background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,0.01)',
            }}>
              {cols.map(c => (
                <td key={c.key} style={{ padding: '10px 12px', color: 'var(--text-primary)', verticalAlign: 'middle' }}>
                  {c.key === 'domain' ? (
                    <span style={{ fontFamily: 'var(--font-display)', fontWeight: 600, textTransform: 'capitalize' }}>
                      {row[c.key]}
                    </span>
                  ) : c.key === 'model' ? (
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-secondary)' }}>
                      {row[c.key]}
                    </span>
                  ) : c.pct ? (
                    <span style={{ fontFamily: 'var(--font-mono)', color: 'var(--accent)' }}>
                      {(row[c.key] * 100).toFixed(1)}%
                    </span>
                  ) : (
                    <span style={{ fontFamily: 'var(--font-mono)' }}>
                      {typeof row[c.key] === 'number' ? row[c.key].toFixed(3) : row[c.key]}
                    </span>
                  )}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function SarcasmComparison({ withSarcasm, withoutSarcasm, gain }) {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 16 }}>
      <ComparisonBlock
        title="Without Sarcasm Handling"
        data={withoutSarcasm}
        color="var(--text-muted)"
        accent="rgba(139,146,160,0.15)"
      />
      <ComparisonBlock
        title="With Sarcasm Handling"
        data={withSarcasm}
        color="var(--positive)"
        accent="var(--positive-dim)"
      />
      <div style={{
        gridColumn: '1 / -1',
        padding: '14px 18px',
        background: gain >= 0 ? 'var(--positive-dim)' : 'var(--negative-dim)',
        border: `1px solid ${gain >= 0 ? 'rgba(62,207,114,0.2)' : 'rgba(245,101,101,0.2)'}`,
        borderRadius: 8,
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
      }}>
        <span style={{ fontSize: 13, color: 'var(--text-primary)', fontFamily: 'var(--font-display)', fontWeight: 600 }}>
          Performance Gain from Sarcasm Handling
        </span>
        <span style={{
          fontFamily: 'var(--font-mono)', fontSize: 18, fontWeight: 700,
          color: gain >= 0 ? 'var(--positive)' : 'var(--negative)',
        }}>
          {gain >= 0 ? '+' : ''}{(gain * 100).toFixed(1)}%
        </span>
      </div>
    </div>
  )
}

function ComparisonBlock({ title, data, color, accent }) {
  return (
    <div style={{
      padding: '16px', background: accent,
      border: `1px solid ${color === 'var(--positive)' ? 'rgba(62,207,114,0.15)' : 'var(--border)'}`,
      borderRadius: 8,
    }}>
      <div style={{ fontSize: 12, fontWeight: 600, color, fontFamily: 'var(--font-display)', marginBottom: 12 }}>
        {title}
      </div>
      {[
        ['Accuracy', (data.accuracy * 100).toFixed(1) + '%'],
        ['Precision', data.macro_precision.toFixed(3)],
        ['Recall', data.macro_recall.toFixed(3)],
        ['F1 Score', data.macro_f1.toFixed(3)],
        ['Weighted F1', data.weighted_f1.toFixed(3)],
      ].map(([k, v]) => (
        <div key={k} style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          padding: '4px 0', borderBottom: '1px solid rgba(255,255,255,0.04)',
        }}>
          <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>{k}</span>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--text-primary)' }}>{v}</span>
        </div>
      ))}
    </div>
  )
}

function OverallMetrics({ data }) {
  const metrics = [
    { label: 'Accuracy', value: (data.accuracy * 100).toFixed(1) + '%', highlight: true },
    { label: 'Macro Precision', value: data.macro_precision.toFixed(3) },
    { label: 'Macro Recall', value: data.macro_recall.toFixed(3) },
    { label: 'Macro F1', value: data.macro_f1.toFixed(3) },
    { label: 'Weighted F1', value: data.weighted_f1.toFixed(3) },
  ]
  return (
    <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
      {metrics.map(m => (
        <div key={m.label} className="card-elevated" style={{ padding: '12px 16px', minWidth: 120 }}>
          <div style={{ fontSize: 10, fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase', marginBottom: 6 }}>
            {m.label}
          </div>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 18, fontWeight: 600, color: m.highlight ? 'var(--accent)' : 'var(--text-primary)' }}>
            {m.value}
          </div>
        </div>
      ))}
    </div>
  )
}

function ConfMatrix({ data }) {
  const { labels, matrix } = data
  const maxVal = Math.max(...matrix.flat())

  return (
    <div>
      <div style={{ marginBottom: 10, fontSize: 12, color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
        Rows: Actual | Columns: Predicted
      </div>
      <table style={{ borderCollapse: 'collapse', fontSize: 13 }}>
        <thead>
          <tr>
            <th style={{ padding: '8px 16px', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)', fontSize: 10 }}></th>
            {labels.map(l => (
              <th key={l} style={{
                padding: '8px 16px', textAlign: 'center',
                fontFamily: 'var(--font-mono)', fontSize: 10,
                color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase',
              }}>Pred {l.slice(0, 3).toUpperCase()}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((row, i) => (
            <tr key={i}>
              <td style={{
                padding: '8px 16px', fontFamily: 'var(--font-mono)', fontSize: 10,
                color: 'var(--text-muted)', letterSpacing: '0.06em', textTransform: 'uppercase',
                whiteSpace: 'nowrap',
              }}>Act {labels[i].slice(0, 3).toUpperCase()}</td>
              {row.map((val, j) => {
                const isCorrect = i === j
                const intensity = maxVal > 0 ? val / maxVal : 0
                return (
                  <td key={j} style={{
                    padding: '10px 20px', textAlign: 'center',
                    fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 600,
                    background: isCorrect
                      ? `rgba(62, 207, 114, ${0.05 + intensity * 0.2})`
                      : val > 0 ? `rgba(245, 101, 101, ${0.03 + intensity * 0.12})` : 'transparent',
                    color: isCorrect ? 'var(--positive)' : val > 0 ? 'var(--negative)' : 'var(--text-muted)',
                    border: '1px solid var(--border)',
                    borderRadius: 4,
                  }}>
                    {val}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
