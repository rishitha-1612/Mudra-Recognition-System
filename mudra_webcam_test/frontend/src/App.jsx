import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import IntroAnimation from "./components/IntroAnimation";
import ModelSelectionPage from "./components/ModelSelectionPage";

const INTRO_DURATION_MS = 5000;

export default function App() {
  const [showIntro, setShowIntro] = useState(true);
  const [launchState, setLaunchState] = useState({
    loading: false,
    method: null,
    message: ""
  });

  useEffect(() => {
    const timer = window.setTimeout(() => setShowIntro(false), INTRO_DURATION_MS);
    return () => window.clearTimeout(timer);
  }, []);

  const runSegmentationScript = async (method) => {
    setLaunchState({
      loading: true,
      method,
      message: `Starting ${method.toUpperCase()}...`
    });

    try {
      const response = await fetch("/api/run-segmentation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ method, camera: 0 })
      });
      const payload = await response.json();
      if (!response.ok || !payload?.ok) {
        throw new Error(payload?.error || "Failed to start segmentation script.");
      }

      setLaunchState({
        loading: false,
        method,
        message: `Running ${method.toUpperCase()} in OpenCV window (PID ${payload.pid}).`
      });
    } catch (error) {
      setLaunchState({
        loading: false,
        method: null,
        message: `Could not start script: ${error.message}`
      });
    }
  };

  return (
    <div className="relative min-h-screen overflow-hidden">
      <AnimatePresence mode="wait">
        {showIntro ? (
          <motion.div
            key="intro"
            initial={{ opacity: 1 }}
            exit={{ opacity: 0, transition: { duration: 0.9, ease: "easeInOut" } }}
            className="absolute inset-0"
          >
            <IntroAnimation />
          </motion.div>
        ) : (
          <motion.div
            key="selector"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.7, ease: "easeOut" }}
            className="min-h-screen"
          >
            <ModelSelectionPage
              onSelect={runSegmentationScript}
              loading={launchState.loading}
              activeMethod={launchState.method}
              message={launchState.message}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
