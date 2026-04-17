import { AnimatePresence, motion } from "framer-motion";

function formatPercent(value) {
  const number = Number(value ?? 0);
  if (Number.isNaN(number)) return "0.0";
  return number.toFixed(1);
}

export default function PredictionPanel({ prediction, confidence, type, top3 }) {
  return (
    <section className="glass-panel rounded-2xl p-5 shadow-glow">
      <div className="flex items-center justify-between gap-3">
        <div className="font-body text-xs uppercase tracking-[0.26em] text-ivory/70">Live Classification</div>
        <div className="flex items-center gap-2 rounded-full border border-ivory/20 bg-[#2a0911]/70 px-3 py-1">
          <span className="relative flex h-2.5 w-2.5">
            <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-[#c9a76a] opacity-75" />
            <span className="relative inline-flex h-2.5 w-2.5 rounded-full bg-[#c9a76a]" />
          </span>
          <span className="font-body text-[11px] tracking-[0.18em] text-ivory/80">LIVE DETECTION ACTIVE</span>
        </div>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={`${prediction}-${confidence}`}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.32 }}
          className="mt-5"
        >
          <h2 className="font-display text-5xl leading-[1.05] tracking-wide text-ivory drop-shadow-[0_0_16px_rgba(201,167,106,0.35)]">
            {prediction}
          </h2>
          <div className="mt-2 font-body text-xl font-semibold text-[#ead7bb]">{formatPercent(confidence)}%</div>
          <div className="mt-2 inline-block rounded-full border border-[#c9a76a]/35 bg-[#4b0f1a]/50 px-4 py-1 font-body text-sm text-ivory/90">
            Type: {type || "Unknown"}
          </div>
        </motion.div>
      </AnimatePresence>

      <div className="mt-6 space-y-3.5">
        {(top3 ?? []).slice(0, 3).map((item, idx) => {
          const score = Number(item?.score ?? 0);
          return (
            <div key={`${item?.label ?? "unknown"}-${idx}`} className="space-y-1.5">
              <div className="flex items-baseline justify-between gap-4">
                <div className="font-body text-sm tracking-wide text-ivory/92">
                  {idx + 1}. {item?.label ?? "Unknown"}
                </div>
                <div className="font-body text-sm text-[#e8d2ae]">{formatPercent(score)}%</div>
              </div>
              <div className="h-2 overflow-hidden rounded-full bg-ivory/12">
                <motion.div
                  className={`h-full rounded-full ${idx === 0 ? "bg-[#c9a76a]" : "bg-[#f1e0c5]/75"}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${Math.max(0, Math.min(100, score))}%` }}
                  transition={{ duration: 0.55, ease: "easeOut" }}
                />
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}
