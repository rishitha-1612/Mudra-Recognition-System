import { AnimatePresence, motion } from "framer-motion";

export default function ImagePanel({ title, imageSrc, accent = "rgba(201,167,106,0.35)", grayscale = false }) {
  return (
    <div className="glass-panel relative overflow-hidden rounded-2xl p-3 shadow-[0_10px_30px_rgba(0,0,0,0.35)]">
      <div className="mb-3 flex items-center justify-between">
        <h3 className="font-display text-2xl tracking-wide text-ivory">{title}</h3>
      </div>

      <div className="relative aspect-video overflow-hidden rounded-xl bg-[rgba(18,2,6,0.8)]">
        <div className="pointer-events-none absolute inset-0" style={{ boxShadow: `inset 0 0 80px ${accent}` }} />
        <AnimatePresence mode="wait">
          {imageSrc ? (
            <motion.img
              key={imageSrc}
              src={imageSrc}
              alt={title}
              className={`h-full w-full object-cover ${grayscale ? "grayscale" : ""}`}
              initial={{ opacity: 0.1, scale: 1.02 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0.2 }}
              transition={{ duration: 0.35, ease: "easeOut" }}
            />
          ) : (
            <motion.div
              key="placeholder"
              className="flex h-full items-center justify-center font-body text-sm tracking-[0.25em] text-ivory/50"
              initial={{ opacity: 0.25 }}
              animate={{ opacity: 0.7 }}
            >
              WAITING FOR LIVE FRAME
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
