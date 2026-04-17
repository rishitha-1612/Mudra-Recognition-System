import { motion } from "framer-motion";
import ImagePanel from "./ImagePanel";
import PredictionPanel from "./PredictionPanel";

export default function LiveDashboard({ data, method, onChangeModel }) {
  const updated = data?.timestamp ? new Date(data.timestamp) : null;
  const formattedTime = updated ? updated.toLocaleTimeString() : "Waiting";
  const methodLabel = (method || "unknown").toUpperCase();

  return (
    <main className="relative mx-auto flex min-h-screen w-full max-w-[1500px] flex-col px-4 pb-7 pt-6 sm:px-6 lg:px-8">
      <header className="mb-5 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
        <div>
          <h1 className="font-display text-4xl tracking-wide text-ivory sm:text-5xl">Mudra Vision</h1>
          <p className="mt-1.5 font-body text-sm uppercase tracking-[0.22em] text-ivory/70">
            Live AI Visualizer for Classical Indian Hand Gestures ({methodLabel})
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="glass-panel rounded-xl px-4 py-2 font-body text-xs uppercase tracking-[0.2em] text-ivory/80">
            Last Update: {formattedTime}
            {data?.fps ? ` | ${data.fps} FPS` : ""}
          </div>
          <button
            type="button"
            onClick={onChangeModel}
            className="rounded-full border border-ivory/30 bg-[#2a0911]/70 px-4 py-2 font-body text-xs uppercase tracking-[0.2em] text-ivory/90 transition hover:border-[#c9a76a]/55 hover:text-[#f1e0c5]"
          >
            Change Model
          </button>
        </div>
      </header>

      {data?.error && (
        <div className="mb-4 rounded-lg border border-[#d49b9b]/35 bg-[#41111b]/65 px-4 py-2 font-body text-sm text-[#f4d8d8]">
          Backend connection issue: {data.error}
        </div>
      )}

      <div className="grid flex-1 gap-5 lg:grid-cols-[1.5fr_1fr]">
        <motion.section
          initial={{ opacity: 0, x: -16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.65, ease: "easeOut" }}
          className="space-y-5"
        >
          <ImagePanel
            title="Original Frame"
            imageSrc={data?.images?.original}
            accent="rgba(201,167,106,0.25)"
          />
          <PredictionPanel
            prediction={data?.prediction}
            confidence={data?.confidence}
            type={data?.type}
            top3={data?.top3}
          />
        </motion.section>

        <motion.section
          initial={{ opacity: 0, x: 16 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.7, ease: "easeOut", delay: 0.05 }}
          className="grid grid-rows-2 gap-5"
        >
          <ImagePanel
            title="Skin Segmentation"
            imageSrc={data?.images?.segmented}
            accent="rgba(188,122,136,0.28)"
          />
          <ImagePanel
            title="Model Input"
            imageSrc={data?.images?.processed}
            accent="rgba(126,111,83,0.35)"
            grayscale
          />
        </motion.section>
      </div>
    </main>
  );
}
