type Point = {
  x: string;
  y: number;
};

type DualSeriesChartProps = {
  leftLabel: string;
  leftSeries: Point[];
  rightLabel: string;
  rightSeries: Point[];
  height?: number;
};

function normalize(values: number[]): { min: number; max: number; range: number } {
  if (values.length === 0) {
    return { min: 0, max: 1, range: 1 };
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = Math.max(1e-9, max - min);
  return { min, max, range };
}

function toPath(values: number[], width: number, height: number, pad: number): string {
  const { max, range } = normalize(values);
  const usableW = width - pad * 2;
  const usableH = height - pad * 2;
  return values
    .map((value, idx) => {
      const x = pad + (idx / Math.max(1, values.length - 1)) * usableW;
      const y = pad + ((max - value) / range) * usableH;
      return `${x},${y}`;
    })
    .join(" ");
}

export function DualSeriesChart({
  leftLabel,
  leftSeries,
  rightLabel,
  rightSeries,
  height = 260,
}: DualSeriesChartProps) {
  const width = 960;
  const leftValues = leftSeries.map((point) => point.y).filter((value) => Number.isFinite(value));
  const rightValues = rightSeries.map((point) => point.y).filter((value) => Number.isFinite(value));

  const leftPath = leftValues.length >= 2 ? toPath(leftValues, width, height, 24) : "";
  const rightPath = rightValues.length >= 2 ? toPath(rightValues, width, height, 24) : "";

  if (!leftPath && !rightPath) {
    return (
      <div className="rounded-xl border border-white/10 bg-black/20 p-4 text-xs text-slate-300">
        Waiting for enough data points to render chart.
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-white/10 bg-black/20 p-2">
      <div className="mb-2 flex flex-wrap items-center gap-3 text-[11px] uppercase tracking-[0.18em] text-slate-300">
        <span className="inline-flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-sky-300" />
          {leftLabel}
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="h-2 w-2 rounded-full bg-emerald-300" />
          {rightLabel}
        </span>
      </div>
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full">
        {[0, 1, 2, 3, 4].map((line) => {
          const y = 24 + (line / 4) * (height - 48);
          return (
            <line
              key={line}
              x1={24}
              x2={width - 24}
              y1={y}
              y2={y}
              stroke="rgba(148,163,184,0.25)"
              strokeDasharray="3 6"
            />
          );
        })}
        {leftPath ? (
          <polyline
            fill="none"
            stroke="#38bdf8"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={leftPath}
          />
        ) : null}
        {rightPath ? (
          <polyline
            fill="none"
            stroke="#34d399"
            strokeWidth="2.2"
            strokeLinecap="round"
            strokeLinejoin="round"
            points={rightPath}
          />
        ) : null}
      </svg>
    </div>
  );
}
