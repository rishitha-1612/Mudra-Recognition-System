import { motion } from "framer-motion";

function LotusButton({ label, onClick, disabled = false }) {
  return (
    <motion.button
      whileHover={disabled ? {} : { scale: 1.01 }}
      whileTap={{ scale: 0.985 }}
      transition={{ duration: 0.25, ease: "easeOut" }}
      onClick={onClick}
      className="lotus-btn group relative w-full overflow-hidden px-7 py-4 text-center font-body text-sm uppercase tracking-[0.2em] text-ivory sm:w-auto"
      type="button"
      disabled={disabled}
    >
      <span className="relative z-10">{label}</span>
    </motion.button>
  );
}

export default function ModelSelectionPage({ onSelect, loading = false, activeMethod = null, message = "" }) {
  return (
    <main className="model-choice-screen relative flex min-h-screen items-center justify-center px-5 py-10">
      <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-[rgba(33,3,9,0.82)] via-[rgba(19,1,6,0.9)] to-[rgba(0,0,0,0.96)]" />

      <motion.section
        initial={{ opacity: 0, y: 14 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.8, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-10 w-full max-w-3xl px-4 py-4 text-center"
      >
        <h1 className="font-display text-5xl tracking-wide text-ivory drop-shadow-[0_8px_30px_rgba(0,0,0,0.5)] sm:text-6xl">
          Mudra Recognition System
        </h1>

        <div className="mx-auto mt-6 max-w-2xl space-y-3 font-body text-[15px] leading-relaxed text-ivory/88 sm:text-base">
          <p>This project detects and interprets hand mudras in real time using computer vision techniques.</p>
          <p>It leverages color space models to accurately segment hand regions from the background.</p>
          <p>
            The system supports multiple detection approaches for improved robustness under different lighting
            conditions.
          </p>
          <p>Users can interact with the webcam to perform gestures and observe live predictions.</p>
          <p>This application demonstrates practical use of image processing in gesture-based interfaces.</p>
          <p>It can be extended for applications in healthcare, dance, and human-computer interaction.</p>
        </div>

        <div className="mx-auto mt-9 flex max-w-xl flex-col items-center justify-center gap-4 sm:flex-row">
          <LotusButton
            label={loading && activeMethod === "ycrcb" ? "Starting..." : "Try YCrCb Model"}
            onClick={() => onSelect("ycrcb")}
            disabled={loading}
          />
          <LotusButton
            label={loading && activeMethod === "hsv" ? "Starting..." : "Try HSV Model"}
            onClick={() => onSelect("hsv")}
            disabled={loading}
          />
        </div>
        {message ? (
          <p className="mx-auto mt-5 max-w-2xl font-body text-sm tracking-wide text-[#f1e0c5]/90">{message}</p>
        ) : null}
      </motion.section>
    </main>
  );
}
