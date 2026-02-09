import { useDashboard } from "../dashboard";
import { formatNumber, formatTime } from "../utils";

export function TradingPage() {
  const { data } = useDashboard();

  return (
    <section className="grid gap-4">
      <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
        <h2 className="font-heading text-lg font-bold text-white">Current Position</h2>
        <div className="mt-3 grid gap-2 text-xs text-slate-200 md:grid-cols-3">
          <p>Symbol: {data.position?.symbol ?? "BTCUSDT"}</p>
          <p>Side: {data.position?.side ?? "flat"}</p>
          <p>Qty: {formatNumber(data.position?.quantity, 6)} BTC</p>
          <p>Avg Entry: {formatNumber(data.position?.avg_entry_price, 2)}</p>
          <p>Unrealized PnL: {formatNumber(data.position?.unrealized_pnl, 2)}</p>
          <p>Realized PnL: {formatNumber(data.position?.realized_pnl, 2)}</p>
        </div>
      </article>

      <div className="grid gap-4 lg:grid-cols-2">
        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <h2 className="font-heading text-lg font-bold text-white">Recent Orders</h2>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-slate-400">
                  <th className="pb-2">Time</th>
                  <th className="pb-2">Side</th>
                  <th className="pb-2">Qty</th>
                  <th className="pb-2">Price</th>
                  <th className="pb-2">Status</th>
                </tr>
              </thead>
              <tbody>
                {data.orders.map((order) => (
                  <tr key={order.id} className="border-t border-white/5 text-slate-200">
                    <td className="py-2">{formatTime(order.created_at)}</td>
                    <td className={order.side === "BUY" ? "py-2 text-mint" : "py-2 text-ember"}>{order.side}</td>
                    <td className="py-2">{formatNumber(order.quantity, 5)}</td>
                    <td className="py-2">{formatNumber(order.price, 2)}</td>
                    <td className="py-2">{order.status}</td>
                  </tr>
                ))}
                {data.orders.length === 0 ? (
                  <tr>
                    <td colSpan={5} className="py-3 text-slate-400">
                      No orders yet.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </article>

        <article className="rounded-2xl border border-white/10 bg-white/5 p-4 shadow-panel backdrop-blur">
          <h2 className="font-heading text-lg font-bold text-white">Recent Fills</h2>
          <div className="mt-3 overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead>
                <tr className="text-slate-400">
                  <th className="pb-2">Time</th>
                  <th className="pb-2">Qty</th>
                  <th className="pb-2">Price</th>
                  <th className="pb-2">Fee</th>
                </tr>
              </thead>
              <tbody>
                {data.fills.map((fill) => (
                  <tr key={fill.id} className="border-t border-white/5 text-slate-200">
                    <td className="py-2">{formatTime(fill.filled_at)}</td>
                    <td className="py-2">{formatNumber(fill.quantity, 5)}</td>
                    <td className="py-2">{formatNumber(fill.price, 2)}</td>
                    <td className="py-2">{formatNumber(fill.fee, 6)}</td>
                  </tr>
                ))}
                {data.fills.length === 0 ? (
                  <tr>
                    <td colSpan={4} className="py-3 text-slate-400">
                      No fills yet.
                    </td>
                  </tr>
                ) : null}
              </tbody>
            </table>
          </div>
        </article>
      </div>
    </section>
  );
}

