import { motion } from "framer-motion";

const outerPetalCount = 32;
const innerPetalCount = 28;
const beadCount = 40;

export default function IntroAnimation() {
  return (
    <div className="relative flex min-h-screen items-center justify-center overflow-hidden bg-[#22070d]">
      <div className="pointer-events-none absolute inset-0 bg-[radial-gradient(circle_at_center,rgba(201,167,106,0.18),transparent_45%)]" />
      <motion.svg
        width="580"
        height="580"
        viewBox="0 0 580 580"
        className="h-[76vmin] w-[76vmin] max-h-[640px] max-w-[640px]"
        initial={{ opacity: 0.3, scale: 0.93 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 1.2 }}
      >
        <g transform="translate(290 290)">
          {[34, 66, 96, 126, 156].map((radius, index) => (
            <motion.circle
              key={radius}
              r={radius}
              stroke="#F3EBDE"
              fill={index === 0 ? "#080808" : "none"}
              strokeWidth={index === 0 ? 2.4 : 1.7}
              strokeOpacity={0.92}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 1, delay: 0.25 + index * 0.2, ease: "easeInOut" }}
            />
          ))}

          {Array.from({ length: innerPetalCount }).map((_, index) => {
            const angle = (360 / innerPetalCount) * index;
            return (
              <motion.g
                key={`inner-${angle}`}
                transform={`rotate(${angle})`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3, delay: 0.9 + index * 0.03 }}
              >
                <motion.path
                  d="M0 -102 Q11 -90 0 -76 Q-11 -90 0 -102 Z"
                  fill="#F7F0E7"
                  stroke="#080808"
                  strokeWidth="1"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.7, delay: 1 + index * 0.03, ease: "easeOut" }}
                />
                <motion.path
                  d="M0 -96 Q4 -90 0 -85 Q-4 -90 0 -96 Z"
                  fill="#080808"
                  stroke="none"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.4, delay: 1.3 + index * 0.03 }}
                />
              </motion.g>
            );
          })}

          {Array.from({ length: beadCount }).map((_, index) => {
            const angle = (360 / beadCount) * index;
            return (
              <motion.g
                key={`bead-${angle}`}
                transform={`rotate(${angle})`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 1.4 + index * 0.02, duration: 0.22 }}
              >
                <circle cx="0" cy="-126" r="4.8" fill="#F7F0E7" />
                <circle cx="0" cy="-113" r="2.3" fill="#F7F0E7" />
              </motion.g>
            );
          })}

          {Array.from({ length: outerPetalCount }).map((_, index) => {
            const angle = (360 / outerPetalCount) * index;
            return (
              <motion.g
                key={`outer-${angle}`}
                transform={`rotate(${angle})`}
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ duration: 0.3, delay: 1.8 + index * 0.02 }}
              >
                <motion.path
                  d="M0 -182 Q15 -160 0 -136 Q-15 -160 0 -182 Z"
                  fill="#F7F0E7"
                  stroke="#080808"
                  strokeWidth="1.35"
                  initial={{ pathLength: 0 }}
                  animate={{ pathLength: 1 }}
                  transition={{ duration: 0.75, delay: 1.8 + index * 0.02 }}
                />
                <path d="M0 -174 Q8 -161 0 -147 Q-8 -161 0 -174 Z" fill="none" stroke="#080808" strokeWidth="1.1" />
                <circle cx="0" cy="-193" r="5.6" fill="#F7F0E7" />
              </motion.g>
            );
          })}

          <motion.circle
            r="20"
            fill="#080808"
            stroke="#F3EBDE"
            strokeWidth="1.5"
            initial={{ opacity: 0.2, scale: 0.85 }}
            animate={{ opacity: [0.35, 1, 0.35], scale: 1 }}
            transition={{ duration: 2.3, repeat: Infinity, ease: "easeInOut" }}
          />
        </g>
      </motion.svg>
      <motion.div
        initial={{ opacity: 0, y: 22 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 2.35, duration: 1 }}
        className="absolute bottom-[8vh] text-center sm:bottom-[7vh]"
      >
        <h1 className="font-display text-6xl tracking-[0.25em] text-ivory sm:text-7xl">Mudra Vision</h1>
        <p className="mt-3 font-body text-sm uppercase tracking-[0.36em] text-[#d9b982]">Live AI Classical Gesture Visualizer</p>
      </motion.div>
    </div>
  );
}
