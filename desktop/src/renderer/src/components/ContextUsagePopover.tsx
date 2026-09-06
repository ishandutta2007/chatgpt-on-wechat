import React, { useCallback, useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'
import { Shrink, Trash2, Sliders } from 'lucide-react'
import { apiClient } from '../api/client'
import { t } from '../i18n'
import { useChatStore } from '../store/chatStore'
import type { ContextUsage } from '../types'
import ContextUsageDonut, {
  ContextMiniPie,
  CONTEXT_SLICE_COLORS,
  formatTokens,
  type ContextSliceKey,
} from './ContextUsageDonut'

const LEGEND: { key: ContextSliceKey; labelKey: string }[] = [
  { key: 'system', labelKey: 'ctx_system' },
  { key: 'tools', labelKey: 'ctx_tools' },
  { key: 'history', labelKey: 'ctx_history' },
  { key: 'free', labelKey: 'ctx_free' },
]

interface ContextUsagePopoverProps {
  sessionId: string
  isStreaming: boolean
  /** Bump this to force a fresh fetch (e.g. when a turn finishes). */
  refreshKey?: number
  onClearContext: () => void
  onAdjust: () => void
  /** Lock/unlock the composer while a synchronous compaction runs. */
  onCompactingChange?: (compacting: boolean) => void
  /** Transient toast (reuses the parent's toast surface). */
  onToast?: (msg: string) => void
}

/**
 * Always-on context pie in the composer, mirroring the web console:
 * - the button shows a mini donut (0% ring when empty),
 * - hover previews the usage card; click pins it (click again to collapse),
 * - the pointer can travel into the card to use its actions,
 * - actions: compact (synchronous, locks input), clear, and adjust budget.
 *
 * Rendered in a body-level portal so the composer's overflow can't clip it.
 */
const ContextUsagePopover: React.FC<ContextUsagePopoverProps> = ({
  sessionId,
  isStreaming,
  refreshKey,
  onClearContext,
  onAdjust,
  onCompactingChange,
  onToast,
}) => {
  const compactContextAction = useChatStore((s) => s.compactContext)
  const [usage, setUsage] = useState<ContextUsage | null>(null)
  const [open, setOpen] = useState(false)
  const [pinned, setPinned] = useState(false)
  const [compacting, setCompacting] = useState(false)
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null)
  const [toast, setToast] = useState('')
  const toastTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Transient toast: prefer the parent's surface, else show a small inline pill.
  const flashToast = useCallback(
    (msg: string) => {
      if (onToast) {
        onToast(msg)
        return
      }
      setToast(msg)
      if (toastTimer.current) clearTimeout(toastTimer.current)
      toastTimer.current = setTimeout(() => setToast(''), 2400)
    },
    [onToast],
  )

  const btnRef = useRef<HTMLButtonElement>(null)
  const cardRef = useRef<HTMLDivElement>(null)
  const hoverTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const hideTimer = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pinnedRef = useRef(false)
  pinnedRef.current = pinned
  const compactingRef = useRef(false)
  compactingRef.current = compacting

  const hasCtx = !!(usage && usage.available && usage.breakdown)

  const fetchUsage = useCallback(async (): Promise<ContextUsage | null> => {
    try {
      const res = await apiClient.getContextUsage(sessionId)
      const ok = res && res.status !== 'error'
      const next = ok ? (res as ContextUsage) : null
      setUsage(next)
      return next
    } catch {
      setUsage(null)
      return null
    }
  }, [sessionId])

  // Prime the mini pie on mount / session change / turn completion.
  useEffect(() => {
    fetchUsage()
  }, [fetchUsage, refreshKey])

  const positionCard = useCallback(() => {
    const el = btnRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    setPos({ x: r.left + r.width / 2, y: r.top - 8 })
  }, [])

  const openCard = useCallback(() => {
    positionCard()
    setOpen(true)
    fetchUsage()
  }, [positionCard, fetchUsage])

  const closeCard = useCallback(() => {
    setPinned(false)
    setOpen(false)
  }, [])

  // --- hover ---
  const onBtnEnter = () => {
    if (hideTimer.current) clearTimeout(hideTimer.current)
    if (hoverTimer.current) clearTimeout(hoverTimer.current)
    hoverTimer.current = setTimeout(openCard, 120)
  }
  const scheduleHide = () => {
    if (hideTimer.current) clearTimeout(hideTimer.current)
    hideTimer.current = setTimeout(() => {
      if (!pinnedRef.current) setOpen(false)
    }, 180)
  }
  const onBtnLeave = () => {
    if (hoverTimer.current) clearTimeout(hoverTimer.current)
    scheduleHide()
  }

  // --- click: toggle pin ---
  const onBtnClick = (e: React.MouseEvent) => {
    e.stopPropagation()
    if (hoverTimer.current) clearTimeout(hoverTimer.current)
    if (hideTimer.current) clearTimeout(hideTimer.current)
    // Second click on a pinned, open card collapses it (unless compacting).
    if (pinned && open && !compacting) {
      closeCard()
      return
    }
    setPinned(true)
    openCard()
  }

  // Dismiss a pinned card on outside click / Esc.
  useEffect(() => {
    if (!pinned) return
    const onDocClick = (e: MouseEvent) => {
      if (compactingRef.current) return
      const target = e.target as Node
      if (cardRef.current?.contains(target) || btnRef.current?.contains(target)) return
      closeCard()
    }
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && !compactingRef.current) closeCard()
    }
    document.addEventListener('click', onDocClick)
    document.addEventListener('keydown', onKey)
    return () => {
      document.removeEventListener('click', onDocClick)
      document.removeEventListener('keydown', onKey)
    }
  }, [pinned, closeCard])

  // Reposition while open (scroll/resize).
  useEffect(() => {
    if (!open) return
    const handler = () => positionCard()
    window.addEventListener('scroll', handler, true)
    window.addEventListener('resize', handler)
    return () => {
      window.removeEventListener('scroll', handler, true)
      window.removeEventListener('resize', handler)
    }
  }, [open, positionCard])

  // --- actions ---
  const doClear = () => {
    if (compacting) return
    if (isStreaming) {
      flashToast(t('ctx_busy_turn'))
      return
    }
    onClearContext()
    setUsage(null)
    closeCard()
  }

  const doAdjust = () => {
    closeCard()
    onAdjust()
  }

  const doCompact = async () => {
    if (compacting) return
    if (isStreaming) {
      flashToast(t('ctx_busy_turn'))
      return
    }
    if (!usage || !usage.available) {
      flashToast(t('ctx_compact_noop'))
      return
    }
    setCompacting(true)
    setPinned(true)
    onCompactingChange?.(true)
    try {
      // The store appends a thread divider on success and returns the result so
      // we can refresh the pie and surface noop/failure toasts here.
      const res = await compactContextAction(sessionId)
      if (!res.ok && !res.noop) {
        flashToast(t('ctx_compact_failed'))
      } else if (res.noop) {
        flashToast(t('ctx_compact_noop'))
      }
      if (res.usage) setUsage(res.usage)
      else await fetchUsage()
    } catch {
      flashToast(t('ctx_compact_failed'))
    } finally {
      setCompacting(false)
      onCompactingChange?.(false)
    }
  }

  const breakdown = usage?.breakdown
  const limit = usage?.limit ?? 0
  const used = usage?.used ?? 0
  const percent = limit > 0 ? Math.min(100, Math.round((used / limit) * 100)) : 0
  const miniSlices = hasCtx
    ? LEGEND.map((l) => ({ key: l.key, value: breakdown![l.key] }))
    : undefined

  return (
    <>
      <span className="relative shrink-0 inline-flex">
        <button
          ref={btnRef}
          type="button"
          onMouseEnter={onBtnEnter}
          onMouseLeave={onBtnLeave}
          onClick={onBtnClick}
          className="shrink-0 w-8 h-8 flex items-center justify-center rounded-btn text-content-secondary hover:text-accent hover:bg-accent-soft cursor-pointer transition-colors"
        >
          <ContextMiniPie slices={miniSlices} />
        </button>
      </span>

      {/* Toast lives in a body portal, centered near the bottom of the window,
          so it can never be covered by the usage card (which sits above the
          button). Rendered above everything, including the pinned card. */}
      {toast &&
        !onToast &&
        createPortal(
          <div
            style={{ position: 'fixed', left: '50%', bottom: 96, transform: 'translateX(-50%)', zIndex: 10001 }}
            className="px-3 py-1.5 rounded-lg text-xs text-white bg-black/85 dark:bg-white/20 shadow-lg whitespace-nowrap pointer-events-none"
          >
            {toast}
          </div>,
          document.body,
        )}

      {open &&
        pos &&
        createPortal(
          <div
            ref={cardRef}
            onMouseEnter={() => {
              if (hideTimer.current) clearTimeout(hideTimer.current)
            }}
            onMouseLeave={scheduleHide}
            style={{
              position: 'fixed',
              left: pos.x,
              top: pos.y,
              transform: 'translate(-50%, -100%)',
              zIndex: 9999,
            }}
            className={`relative rounded-xl bg-elevated border border-default shadow-xl ${
              hasCtx ? 'w-[248px] p-3' : 'px-3 py-2'
            }`}
          >
            <div className="flex items-baseline justify-between mb-2">
              <span className="text-[12px] font-medium text-content">{t('ctx_usage_title')}</span>
              {usage?.estimated && hasCtx && (
                <span className="text-[10px] text-content-tertiary">{t('ctx_estimated')}</span>
              )}
            </div>

            {!hasCtx ? (
              <div className="text-[11px] text-content-tertiary text-center whitespace-nowrap">
                {usage === null ? t('ctx_error') : t('ctx_empty')}
              </div>
            ) : (
              <>
                <div className="flex justify-center mb-2">
                  <ContextUsageDonut
                    percent={percent}
                    slices={LEGEND.map((l) => ({ key: l.key, value: breakdown![l.key] }))}
                  />
                </div>
                <div className="space-y-1">
                  {LEGEND.map((l) => (
                    <div key={l.key} className="flex items-center gap-1.5 text-[11px]">
                      <span
                        className="w-2 h-2 rounded-sm shrink-0"
                        style={{ background: CONTEXT_SLICE_COLORS[l.key] }}
                      />
                      <span className="text-content-secondary flex-1 truncate">{t(l.labelKey)}</span>
                      <span className="text-content tabular-nums">{formatTokens(breakdown![l.key])}</span>
                    </div>
                  ))}
                </div>
                <div className="mt-2 pt-2 border-t border-default text-[10px] text-content-tertiary tabular-nums">
                  {t('ctx_used_of')
                    .replace('{used}', formatTokens(used))
                    .replace('{limit}', formatTokens(limit))}
                </div>

                {/* Actions */}
                <div className="mt-2.5 pt-2.5 border-t border-default flex gap-1.5">
                  <button
                    type="button"
                    onClick={doCompact}
                    className="flex-1 flex items-center justify-center gap-1 px-1 py-1.5 rounded-lg border border-default bg-surface text-[11px] text-content-secondary hover:bg-surface-2 hover:text-content cursor-pointer transition-colors"
                  >
                    <Shrink size={12} />
                    <span>{t('ctx_act_compact')}</span>
                  </button>
                  <button
                    type="button"
                    onClick={doClear}
                    className="flex-1 flex items-center justify-center gap-1 px-1 py-1.5 rounded-lg border border-default bg-surface text-[11px] text-content-secondary hover:bg-surface-2 hover:text-content cursor-pointer transition-colors"
                  >
                    <Trash2 size={12} />
                    <span>{t('ctx_act_clear')}</span>
                  </button>
                  <button
                    type="button"
                    onClick={doAdjust}
                    className="flex-1 flex items-center justify-center gap-1 px-1 py-1.5 rounded-lg border border-default bg-surface text-[11px] text-content-secondary hover:bg-surface-2 hover:text-content cursor-pointer transition-colors"
                  >
                    <Sliders size={12} />
                    <span>{t('ctx_act_adjust')}</span>
                  </button>
                </div>
              </>
            )}

            {/* Loading overlay during synchronous compaction. */}
            {compacting && (
              <div className="absolute inset-0 rounded-xl bg-elevated/85 backdrop-blur-[1px] flex flex-col items-center justify-center gap-2 text-[12px] text-content">
                <span className="w-5 h-5 rounded-full border-2 border-border-strong border-t-accent animate-spin" />
                <span>{t('ctx_compacting')}</span>
              </div>
            )}
          </div>,
          document.body,
        )}
    </>
  )
}

export default ContextUsagePopover
