import { useEffect, useState } from 'react'
import EditorModule from 'react-simple-code-editor'
import Prism from 'prismjs'
import 'prismjs/components/prism-python'
import { EXAMPLES } from './examples'

const Editor = EditorModule.default ?? EditorModule   // CommonJS interop (Vite dev vs build)

// Python grammar + traceback-specific tokens
Prism.languages.traceback = Prism.languages.extend('python', {
  'tb-head': /^Traceback \(most recent call last\):/m,
  'tb-error': /^[\w.]*(?:Error|Exception|Interrupt|Exit|Warning|Detected|Cancelled)\b.*$/m,
})
const highlight = (code) => Prism.highlight(code, Prism.languages.traceback, 'traceback')

const Spinner = () => (
  <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-white/30 border-t-white" />
)

function AttentionView({ tokens, weights }) {
  return (
    <div className="mt-5 rounded-xl border border-white/10 bg-black/30 p-4 font-mono text-xs leading-7">
      {tokens.map((t, i) =>
        t === '<NL>' ? <br key={i} /> : (
          <span key={i} className="mr-1 rounded px-1"
                style={{ background: `rgba(34,211,238,${Math.min(0.85, (weights[i] || 0) * 0.85)})`,
                         color: (weights[i] || 0) > 0.55 ? '#021014' : '#cbd5e1' }}>
            {t}
          </span>
        ))}
    </div>
  )
}

export default function App() {
  const [text, setText] = useState(EXAMPLES[0].text)
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [copied, setCopied] = useState(false)
  const [showAtt, setShowAtt] = useState(false)
  const [info, setInfo] = useState(null)

  function checkHealth() {
    setInfo(null)
    fetch('/api/health')
      .then(async (r) => { const j = await r.json().catch(() => ({})); setInfo(r.ok ? j : { status: 'error', detail: j.detail || `Backend error ${r.status}` }) })
      .catch(() => setInfo({ status: 'offline' }))
  }
  useEffect(checkHealth, [])

  async function generate() {
    setLoading(true); setError(''); setCopied(false)
    try {
      const r = await fetch('/api/summarize', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ traceback: text }),
      })
      if (!r.ok) {
        const d = (await r.json().catch(() => ({}))).detail
        throw new Error(typeof d === 'string' ? d : `The request was rejected (HTTP ${r.status}).`)
      }
      setResult(await r.json())
    } catch (e) {
      setResult(null)
      setError(e instanceof TypeError ? 'Cannot reach the backend. Start it with: python -m uvicorn webapp.backend.main:app --port 8000' : e.message)
      checkHealth()
    } finally { setLoading(false) }
  }

  function copy() {
    navigator.clipboard.writeText(result.summary).then(() => { setCopied(true); setTimeout(() => setCopied(false), 1500) })
  }

  const online = info && info.status === 'ok'

  return (
    <div className="relative min-h-screen overflow-hidden">
      <div className="blob pointer-events-none absolute -left-32 -top-32 h-96 w-96 rounded-full bg-indigo-600/25 blur-3xl" />
      <div className="blob pointer-events-none absolute -right-32 top-1/3 h-96 w-96 rounded-full bg-cyan-500/20 blur-3xl" style={{ animationDelay: '-6s' }} />

      <main className="relative mx-auto max-w-5xl px-4 py-10 sm:py-14">
        <header className="fade-up text-center">
          <h1 className="bg-gradient-to-r from-indigo-300 via-cyan-300 to-emerald-300 bg-clip-text text-4xl font-extrabold tracking-tight text-transparent sm:text-5xl">
            CodeNarrator
          </h1>
          <p className="mx-auto mt-3 max-w-xl text-slate-400">
            Paste a Python traceback and get a short English summary from an LSTM encoder–decoder with attention, trained from scratch.
          </p>
          <div className="mt-4 inline-flex items-center gap-2 rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-400">
            <span className={`h-2 w-2 rounded-full ${online ? 'bg-emerald-400' : 'bg-rose-500'}`} />
            {online ? `Model loaded · ${(info.parameters / 1e6).toFixed(1)}M params · ${info.device}`
              : info === null ? 'Connecting…' : info.status === 'error' ? `Model unavailable: ${info.detail}` : 'Backend offline'}
            {!online && info !== null && <button onClick={checkHealth} className="ml-1 underline hover:text-white">Retry</button>}
          </div>
        </header>

        <section className="fade-up mt-8 rounded-2xl border border-white/10 bg-white/[0.04] p-4 shadow-2xl backdrop-blur sm:p-6">
          <div className="mb-3 flex flex-wrap items-center gap-2">
            <span className="mr-1 text-xs uppercase tracking-wider text-slate-500">Examples</span>
            {EXAMPLES.map((ex) => (
              <button key={ex.label} onClick={() => { setText(ex.text); setResult(null); setError('') }}
                      className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-slate-300 transition hover:border-cyan-400/50 hover:text-cyan-200">
                {ex.label}
              </button>
            ))}
          </div>

          <div className="max-h-80 overflow-auto rounded-xl border border-white/10 bg-[#0b1020] focus-within:border-indigo-400/60">
            <Editor value={text} onValueChange={setText} highlight={highlight} padding={16}
                    onKeyDown={(e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && text.trim() && !loading) generate() }}
                    placeholder="Paste a Python traceback here…"
                    style={{ fontFamily: 'ui-monospace, SFMono-Regular, Menlo, Consolas, monospace', fontSize: 13, minHeight: 200 }}
                    textareaClassName="focus:outline-none" />
          </div>

          <div className="mt-4 flex flex-wrap items-center gap-3">
            <button onClick={generate} disabled={loading || !text.trim()}
                    className="inline-flex items-center gap-2 rounded-xl bg-gradient-to-r from-indigo-500 to-cyan-500 px-6 py-2.5 font-semibold text-white shadow-lg shadow-indigo-900/40 transition hover:brightness-110 active:scale-95 disabled:cursor-not-allowed disabled:opacity-50">
              {loading ? <><Spinner /> Generating…</> : 'Generate Summary'}
            </button>
            <button onClick={() => { setText(''); setResult(null); setError('') }}
                    className="rounded-xl border border-white/10 px-4 py-2.5 text-sm text-slate-300 transition hover:bg-white/10">
              Clear
            </button>
            <span className="ml-auto text-xs text-slate-500">Ctrl+Enter to generate · {text.length.toLocaleString()} chars</span>
          </div>
        </section>

        {error && (
          <div className="fade-up mt-6 rounded-xl border border-rose-500/30 bg-rose-500/10 p-4 text-sm text-rose-200">{error}</div>
        )}

        {result && (
          <section key={result.summary + result.latency_ms} className="fade-up mt-6 rounded-2xl border border-cyan-400/20 bg-gradient-to-br from-indigo-500/10 to-cyan-500/10 p-5 shadow-2xl sm:p-7">
            <div className="flex items-center justify-between">
              <h2 className="text-xs font-semibold uppercase tracking-widest text-cyan-300">Generated summary</h2>
              <button onClick={copy} className="rounded-lg border border-white/10 px-3 py-1 text-xs text-slate-300 transition hover:bg-white/10">
                {copied ? 'Copied ✓' : 'Copy'}
              </button>
            </div>
            <p className="mt-3 text-xl font-medium leading-relaxed text-white sm:text-2xl">{result.summary || '(empty output)'}</p>

            {result.warnings && result.warnings.length > 0 && (
              <ul className="mt-4 space-y-1 rounded-xl border border-amber-400/30 bg-amber-400/10 p-3 text-sm text-amber-200">
                {result.warnings.map((w) => <li key={w}>⚠ {w}</li>)}
              </ul>
            )}

            <div className="mt-5 flex flex-wrap gap-2 text-xs text-slate-400">
              <span className="rounded-full bg-white/5 px-3 py-1">{result.latency_ms} ms</span>
              <span className="rounded-full bg-white/5 px-3 py-1">{result.source_tokens.length} input tokens</span>
              <span className={`rounded-full px-3 py-1 ${result.n_unk > 0 ? 'bg-amber-500/15 text-amber-300' : 'bg-white/5'}`}>
                {result.n_unk} unknown tokens
              </span>
              {result.truncated && <span className="rounded-full bg-amber-500/15 px-3 py-1 text-amber-300">input truncated</span>}
            </div>

            <button onClick={() => setShowAtt(!showAtt)} className="mt-4 text-xs text-cyan-300 hover:underline">
              {showAtt ? 'Hide' : 'Show'} attention highlights
            </button>
            {showAtt && <AttentionView tokens={result.source_tokens} weights={result.attention} />}
          </section>
        )}

        <footer className="mt-10 text-center text-xs leading-relaxed text-slate-500">
          Trained on a synthetic dataset of 104 error families (not real tracebacks). Summaries for tracebacks outside those families are often fluent but wrong —
          treat the output as a demo of the model, not as a reliable diagnosis.
        </footer>
      </main>
    </div>
  )
}
