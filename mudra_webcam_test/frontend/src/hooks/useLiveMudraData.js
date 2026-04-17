import { useEffect, useMemo, useRef, useState } from "react";

function toImageSource(value) {
  if (!value) return null;
  if (value.startsWith("data:image")) return value;
  if (value.startsWith("http://") || value.startsWith("https://")) return value;
  const raw = value.trim();
  if (raw.startsWith("/")) return raw;
  return `data:image/jpeg;base64,${raw}`;
}

const emptyState = {
  prediction: "Awaiting detection",
  confidence: 0,
  type: "Unknown",
  top3: [],
  images: {
    original: null,
    segmented: null,
    processed: null
  },
  timestamp: null,
  fps: null,
  error: null
};

export function useLiveMudraData(endpoint = "/api/live-prediction", intervalMs = 500) {
  const [state, setState] = useState(emptyState);
  const inFlight = useRef(false);

  useEffect(() => {
    if (!endpoint) {
      setState(emptyState);
      return undefined;
    }

    let mounted = true;
    const controller = new AbortController();

    const fetchLive = async () => {
      if (inFlight.current) return;
      inFlight.current = true;
      try {
        const response = await fetch(endpoint, {
          method: "GET",
          headers: { Accept: "application/json" },
          signal: controller.signal
        });
        if (!response.ok) {
          throw new Error(`Backend returned ${response.status}`);
        }

        const payload = await response.json();
        if (!mounted) return;

        setState({
          prediction: payload?.prediction ?? "Awaiting detection",
          confidence: Number(payload?.confidence ?? 0),
          type: payload?.type ?? "Unknown",
          top3: Array.isArray(payload?.top3) ? payload.top3 : [],
          images: {
            original: toImageSource(payload?.images?.original),
            segmented: toImageSource(payload?.images?.segmented),
            processed: toImageSource(payload?.images?.processed)
          },
          timestamp: payload?.timestamp ?? new Date().toISOString(),
          fps: payload?.fps ?? null,
          error: null
        });
      } catch (error) {
        if (!mounted || error?.name === "AbortError") return;
        setState((prev) => ({ ...prev, error: error.message }));
      } finally {
        inFlight.current = false;
      }
    };

    fetchLive();
    const intervalId = window.setInterval(fetchLive, intervalMs);

    return () => {
      mounted = false;
      controller.abort();
      window.clearInterval(intervalId);
    };
  }, [endpoint, intervalMs]);

  return useMemo(() => state, [state]);
}
