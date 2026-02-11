import type { ReactNode } from "react";

type PanelProps = {
  title?: string;
  subtitle?: string;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
};

export function Panel({ title, subtitle, right, children, className = "" }: PanelProps) {
  return (
    <article
      className={`rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur ${className}`}
    >
      {title ? (
        <header className="mb-3 flex items-start justify-between gap-3">
          <div>
            <h2 className="font-heading text-lg font-bold text-white">{title}</h2>
            {subtitle ? <p className="mt-1 text-xs text-slate-300">{subtitle}</p> : null}
          </div>
          {right}
        </header>
      ) : null}
      {children}
    </article>
  );
}
