import React from 'react'

/** Slice colors, shared by the donut and its legend so they can't drift.
 *  CSS variables, so dark mode follows the theme without extra work. */
export const CONTEXT_SLICE_COLORS = {
  system: 'var(--accent-active)',
  tools: 'var(--accent)',
  history: 'var(--warning)',
  free: 'var(--border-strong)',
} as const

export type ContextSliceKey = keyof typeof CONTEXT_SLICE_COLORS

/** 18240 -> "18.2k". Keeps legend rows narrow enough to stay on one line. */
export function formatTokens(n: number): string {
  if (n < 1000) return String(n)
  if (n < 10000) return `${(n / 1000).toFixed(1)}k`
  return `${Math.round(n / 1000)}k`
}

interface ContextUsageDonutProps {
  /** Token count per slice, drawn in this order. */
  slices: { key: ContextSliceKey; value: number }[]
  /** Percentage shown in the hole, 0-100. */
  percent: number
  size?: number
}

/**
 * Context-usage donut, drawn as stroked arcs on concentric circles — a pie of
 * four slices does not justify pulling in a charting library.
 *
 * Each slice is one <circle> whose dasharray exposes just its own share of the
 * circumference, offset by everything before it. The group is rotated -90deg so
 * the first slice starts at 12 o'clock.
 */
const ContextUsageDonut: React.FC<ContextUsageDonutProps> = ({ slices, percent, size = 96 }) => {
  const stroke = 12
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const total = slices.reduce((sum, s) => sum + Math.max(0, s.value), 0)

  let offset = 0
  const arcs = slices.map((s) => {
    const fraction = total > 0 ? Math.max(0, s.value) / total : 0
    const arc = { ...s, fraction, start: offset }
    offset += fraction
    return arc
  })

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
      <g transform={`rotate(-90 ${size / 2} ${size / 2})`}>
        {arcs.map(
          (a) =>
            a.fraction > 0 && (
              <circle
                key={a.key}
                cx={size / 2}
                cy={size / 2}
                r={r}
                fill="none"
                stroke={CONTEXT_SLICE_COLORS[a.key]}
                strokeWidth={stroke}
                strokeDasharray={`${a.fraction * c} ${c}`}
                strokeDashoffset={-a.start * c}
              />
            ),
        )}
      </g>
      <text
        x="50%"
        y="50%"
        textAnchor="middle"
        dominantBaseline="central"
        className="fill-content text-[15px] font-medium"
      >
        {percent}%
      </text>
    </svg>
  )
}

export default ContextUsageDonut

/**
 * Tiny always-on donut for the composer button (16px, thin stroke). Shows the
 * fill/percent at a glance without opening the card. An empty session is a
 * faint full ring — the button always reads as a pie, never flips to an icon.
 */
export const ContextMiniPie: React.FC<{
  slices?: { key: ContextSliceKey; value: number }[]
  size?: number
}> = ({ slices, size = 16 }) => {
  const stroke = 4
  const r = (size - stroke) / 2
  const c = 2 * Math.PI * r
  const list = slices || []
  const total = list.reduce((sum, s) => sum + Math.max(0, s.value), 0)

  if (total <= 0) {
    return (
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="currentColor" strokeWidth={stroke} opacity={0.35} />
      </svg>
    )
  }

  let offset = 0
  const arcs = list.map((s) => {
    const fraction = Math.max(0, s.value) / total
    const arc = { ...s, fraction, start: offset }
    offset += fraction
    return arc
  })

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} className="shrink-0">
      {/* faint track */}
      <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="currentColor" strokeWidth={stroke} opacity={0.18} />
      <g transform={`rotate(-90 ${size / 2} ${size / 2})`}>
        {arcs.map(
          (a) =>
            a.fraction > 0 &&
            a.key !== 'free' && (
              <circle
                key={a.key}
                cx={size / 2}
                cy={size / 2}
                r={r}
                fill="none"
                stroke={CONTEXT_SLICE_COLORS[a.key]}
                strokeWidth={stroke}
                strokeDasharray={`${a.fraction * c} ${c}`}
                strokeDashoffset={-a.start * c}
              />
            ),
        )}
      </g>
    </svg>
  )
}
