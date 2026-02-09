type SparklineProps = {
  points: number[];
};

export function Sparkline({ points }: SparklineProps) {
  if (points.length < 2) {
    return <div className="text-xs text-slate-400">No equity data yet.</div>;
  }

  const min = Math.min(...points);
  const max = Math.max(...points);
  const range = max - min || 1;
  const mapped = points.map((value, index) => {
    const x = (index / (points.length - 1)) * 100;
    const y = 100 - ((value - min) / range) * 100;
    return `${x},${y}`;
  });

  return (
    <svg viewBox="0 0 100 100" className="h-28 w-full overflow-visible">
      <polyline
        fill="none"
        stroke="url(#sparklineGradient)"
        strokeLinecap="round"
        strokeLinejoin="round"
        strokeWidth="2.2"
        points={mapped.join(" ")}
      />
      <defs>
        <linearGradient id="sparklineGradient" x1="0%" x2="100%" y1="0%" y2="0%">
          <stop offset="0%" stopColor="#2dd4bf" />
          <stop offset="100%" stopColor="#fb923c" />
        </linearGradient>
      </defs>
    </svg>
  );
}

