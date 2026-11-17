"""Builds docs/index.html: a mobile-friendly live dashboard (Plotly via CDN)."""
from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go

COLORS = {"Logistic regression": "#2a78d6", "LightGBM": "#eb6834", "Ensemble": "#1baf7a",
          "Momentum rule": "#eda100", "Equal-weight (all)": "#e87ba4", "Earnings drift rule": "#008300",
          "S&P 500 tracker (£)": "#52514e",
          # "+ earnings" versions share their base model's colour, drawn dotted
          "LR + earnings": "#2a78d6", "LightGBM + earnings": "#eb6834", "Ensemble + earnings": "#1baf7a"}
DASH = {"S&P 500 tracker (£)": "dash", "LR + earnings": "dot", "LightGBM + earnings": "dot", "Ensemble + earnings": "dot"}
TWIN = {"Logistic regression": "LR + earnings", "LightGBM": "LightGBM + earnings", "Ensemble": "Ensemble + earnings"}
TWIN.update({v: k for k, v in TWIN.items()})
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
CONFIG = {"displayModeBar": False, "responsive": True}


def _style(fig, title, h=320, two_col_legend=False):
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=15, color=INK)), height=h, template="plotly_white",
                      font=dict(family="system-ui,-apple-system,Segoe UI,sans-serif", color=MUTED, size=12),
                      margin=dict(l=48, r=12, t=48, b=30), hovermode="x unified", paper_bgcolor=SURF,
                      plot_bgcolor=SURF, legend=dict(orientation="h", y=-0.18, font=dict(size=11)))
    fig.update_xaxes(gridcolor=GRID, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, linecolor=GRID)
    if two_col_legend:
        fig.update_layout(legend=dict(orientation="h", y=-0.1, yanchor="top", entrywidth=0.5,
                                      entrywidthmode="fraction", font=dict(size=10.5)))
    fig.update_layout(autosize=True)
    return fig.to_html(full_html=False, include_plotlyjs=False, config=CONFIG,
                       default_width="100%", default_height=f"{h}px")


def _pct(x, sign=True):
    return "-" if x is None else (f"{x:+.2%}" if sign else f"{x:.1%}")


def _table(rows, cols):
    head = "".join(f"<th>{escape(c)}</th>" for c, _ in cols)
    body = "".join("<tr>" + "".join(f"<td>{escape(str(fn(r)))}</td>" for _, fn in cols) + "</tr>" for r in rows)
    return f"<div class='tw'><table class='tbl'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def build_dashboard(results: dict, bt: dict, nav: pd.DataFrame) -> str:
    p, v = results["paper"], results.get("verdict") or {}
    model = results.get("model") or "-"
    ahead = p["value_gbp"] - p["sp500_value_gbp"]

    # --- live paper chart
    if len(nav) >= 2:
        f = go.Figure()
        f.add_trace(go.Scatter(x=nav["date"], y=nav["portfolio_gbp"], name="My paper portfolio", mode="lines+markers",
                               line=dict(color=COLORS.get(model, "#2a78d6"), width=2.5), marker=dict(size=6),
                               hovertemplate="£%{y:,.2f}"))
        f.add_trace(go.Scatter(x=nav["date"], y=nav["sp500_gbp"], name="S&P 500 tracker", mode="lines+markers",
                               line=dict(color=COLORS["S&P 500 tracker (£)"], width=1.8, dash="dash"), marker=dict(size=6),
                               hovertemplate="£%{y:,.2f}"))
        f.update_yaxes(tickprefix="£"); f.update_xaxes(type="date", tickformat="%d %b")
        live_chart = _style(f, "Paper portfolio vs S&P 500 (live, £)", 300)
    else:
        live_chart = "<p class='muted'>The live chart appears after the second daily run.</p>"

    # --- backtest charts
    bt_html = ""
    if bt and bt.get("growth"):
        key = [model, TWIN.get(model), "Earnings drift rule", "Equal-weight (all)", "S&P 500 tracker (£)"]
        key = [k for k in dict.fromkeys(key) if k and k in bt["growth"]]
        g = go.Figure()
        for s, series in bt["growth"].items():
            g.add_trace(go.Scatter(x=list(series), y=list(series.values()), name=s,
                                   visible=True if s in key else "legendonly",       # tap legend to show others
                                   line=dict(color=COLORS.get(s, "#888"), width=2.5 if s == model else 1.6,
                                             dash=DASH.get(s, "solid")), hovertemplate="£%{y:,.0f}"))
        g.update_yaxes(type="log", tickprefix="£")
        bt_html += f"<div class='card'>{_style(g, '£100 invested · backtest, real S&P 500', 470, two_col_legend=True)}</div>"
        years = sorted({y for s in bt["yearly"].values() for y in s})
        strat = key
        short = {"Logistic regression": "LogReg", "LightGBM": "LGBM", "Ensemble": "Ens", "Momentum rule": "Mom",
                 "Equal-weight (all)": "EqW", "S&P 500 tracker (£)": "S&P", "LR + earnings": "LR+E",
                 "LightGBM + earnings": "LGBM+E", "Ensemble + earnings": "Ens+E", "Earnings drift rule": "EarnRule"}
        z = [[bt["yearly"][s].get(y) for s in strat] for y in years]          # years = rows (phone friendly)
        hm = go.Figure(go.Heatmap(z=z, y=years, x=[short.get(s, s) for s in strat], zmid=0, zmin=-0.4, zmax=0.4,
                                  colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]],
                                  text=[[("" if c is None else f"{c:+.0%}") for c in r] for r in z], texttemplate="%{text}",
                                  hovertemplate="%{x}, %{y}: %{text}<extra></extra>", showscale=False))
        hm.update_layout(hovermode="closest")
        hm.update_yaxes(autorange="reversed", type="category"); hm.update_xaxes(side="top")
        bt_html += f"<div class='card'>{_style(hm, 'Return each year (blue = gain, red = loss)', 120 + 26 * len(years))}</div>"
        rows = sorted(bt["table"].items(), key=lambda kv: -kv[1]["Sharpe"])
        bt_html += "<div class='card'>" + _table(rows, [
            ("Strategy", lambda r: r[0]), ("Sharpe", lambda r: f"{r[1]['Sharpe']:.2f}"),
            ("CAGR", lambda r: f"{r[1]['CAGR']:.1%}"), ("Worst fall", lambda r: f"{r[1]['Max drawdown']:.0%}"),
            ("£100 →", lambda r: f"£{r[1]['£100 grew to']:,.0f}")]) + "</div>"
        et = bt.get("earnings_test")
        if et:
            better = sum(pr["sharpe_with"] > pr["sharpe_without"] for pr in et["pairs"])
            bt_html += ("<h2>🧾 Earnings test: do earnings surprises help?</h2><div class='card'>" + _table(et["pairs"], [
                ("Model", lambda r: r["model"]), ("Sharpe without", lambda r: f"{r['sharpe_without']:.2f}"),
                ("Sharpe with", lambda r: f"{r['sharpe_with']:.2f}"),
                ("AUC without → with", lambda r: f"{r['auc_without']:.3f} → {r['auc_with']:.3f}"),
                ("CAGR without → with", lambda r: f"{r['cagr_without']:.1%} → {r['cagr_with']:.1%}")]) +
                f"<p class='muted'>Earnings data covers {et['coverage']:.0%} of stock-days. Adding earnings improved Sharpe for "
                f"{better} of 3 models. Simple earnings drift rule (buy the strongest reactions to results in the last ~3 months): "
                f"Sharpe {et['drift_rule']['Sharpe']:.2f}, CAGR {et['drift_rule']['CAGR']:.1%}.</p></div>")
        auc = ", ".join(f"{k} {val:.3f}" for k, val in (bt.get("auc") or {}).items())
        cov = bt.get("coverage") or {}
        cov_txt = f"{min(cov.values()):.0%}–{max(cov.values()):.0%}" if cov else "-"
        bt_html += (f"<p class='muted'>Backtest {escape(' → '.join(bt.get('period') or []))}, refreshed {escape(str(bt.get('generated')))}. "
                    f"AUC (0.50 = coin flip): {escape(auc)}. Data coverage of S&P members: {cov_txt}.</p>")

    kpi = lambda label, value, sub="", cls="": (f"<div class='kpi {cls}'><div class='kl'>{escape(label)}</div>"
                                               f"<div class='kv'>{escape(value)}</div><div class='ks'>{escape(sub)}</div></div>")
    kpis = "".join([
        kpi("Paper portfolio", f"£{p['value_gbp']:,.2f}", f"{_pct(p['return_since_start'])} since {p['start_date']}"),
        kpi("S&P 500 tracker", f"£{p['sp500_value_gbp']:,.2f}", f"{_pct(p['sp500_return_since_start'])} since start"),
        kpi("vs S&P 500", f"{'+' if ahead >= 0 else '-'}£{abs(ahead):,.2f}", "ahead" if ahead >= 0 else "behind",
            "pos" if ahead >= 0 else "neg"),
        kpi("Today", _pct(p.get("day_change")), f"S&P {_pct(p.get('sp500_day_change'))}"),
        kpi("Next rebalance", f"{results['next_rebalance_in_trading_days']} trading days",
            "rebalanced today ✅" if results["rebalanced_today"] else f"last: {results['last_rebalance']['date']}"),
        kpi("Model", model, (f"backtest Sharpe {v['model_sharpe']:.2f} vs S&P {v['sp500_sharpe']:.2f}" if v else "")),
    ])
    holdings = _table(results["holdings"], [
        ("Stock", lambda h: h["ticker"]), ("Value", lambda h: f"£{h['value_gbp']:.2f}"),
        ("Since bought", lambda h: _pct(h["return_since_entry"])), ("Shares", lambda h: f"{h['shares']:.4f}"),
        ("Sector", lambda h: h["sector"])])
    icon = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "KEEP": "⚪ KEEP"}
    trades = _table(results["last_rebalance"]["trades"], [
        ("Stock", lambda t: t["ticker"]), ("Action", lambda t: icon.get(t["action"], "🔵 " + t["action"])),
        ("Change", lambda t: f"£{t['change_gbp']:+.2f}"), ("Why", lambda t: t["why"])])
    verdict = ""
    if v:
        ok = v["beats_sp500"]
        verdict = (f"<div class='note {'pos' if ok else 'neg'}'>Backtest verdict: <b>{escape(model)}</b> "
                   f"{'beats' if ok else 'does <b>not</b> beat'} a plain S&P 500 tracker on risk-adjusted return "
                   f"(Sharpe {v['model_sharpe']:.2f} vs {v['sp500_sharpe']:.2f}). Treat this as a live experiment.</div>")
    warn = "".join(f"<li>{escape(w)}</li>" for w in results.get("warnings", []))
    warn = f"<div class='note neg'><b>Warnings</b><ul>{warn}</ul></div>" if warn else ""

    css = """
:root{color-scheme:light}*{box-sizing:border-box}
body{margin:0;background:#f5f4f0;font-family:system-ui,-apple-system,'Segoe UI',sans-serif;color:#0b0b0b}
.wrap{max-width:1100px;margin:0 auto;padding:18px 14px 50px}
h1{font-size:22px;margin:0 0 2px}h2{font-size:17px;margin:26px 0 8px}
.muted{color:#52514e;font-size:12.5px;line-height:1.45}
.kpis{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(150px,100%),1fr));gap:10px;margin-top:14px}
.kpi{background:#fcfcfb;border:1px solid #e6e5e0;border-radius:12px;padding:12px}
.kl{font-size:12px;color:#52514e}.kv{font-size:21px;font-weight:650;margin:3px 0}.ks{font-size:12px;color:#52514e}
.kpi.pos .kv{color:#006300}.kpi.neg .kv{color:#b42318}
.card{min-width:0;background:#fcfcfb;border:1px solid #e6e5e0;border-radius:12px;padding:10px;margin-top:10px}
.grid2{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:10px}.grid2>*{min-width:0}
@media(max-width:820px){.grid2{grid-template-columns:minmax(0,1fr)}}
.tw{overflow-x:auto}.tbl{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
.tbl th,.tbl td{padding:7px 8px;border-bottom:1px solid #e6e5e0;text-align:right;white-space:nowrap}
.tbl th:first-child,.tbl td:first-child{text-align:left;font-weight:600}.tbl thead th{color:#52514e;font-weight:600}
.note{border-radius:12px;padding:11px 13px;font-size:13px;margin-top:12px;border:1px solid}
.note.pos{background:#eef8ee;border-color:#bfe3bf}.note.neg{background:#fff4f2;border-color:#f3c9c2}
.foot{margin-top:28px;font-size:12px;color:#52514e}
"""
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">
<title>Trading System · {escape(results['market_date'])}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>{css}</style></head><body><div class="wrap">
<h1>📈 Trading System</h1>
<div class="muted">Prices to {escape(results['market_date'])} · updated {escape(results['generated_at_utc'])} UTC ·
£1 = ${results['gbpusd']:.3f} · £100 paper portfolio · picks from the real S&P 500, rebalanced monthly</div>
<div class="kpis">{kpis}</div>{verdict}{warn}
<h2>📈 Live paper trading</h2><div class="card">{live_chart}</div>
<div class="grid2"><div><h2>💼 Holdings</h2><div class="card">{holdings}</div></div>
<div><h2>🔁 Last rebalance ({escape(results['last_rebalance']['date'])})</h2><div class="card">{trades}</div></div></div>
<h2>📊 Backtest (no hindsight)</h2><p class="muted">Tip: tap names in a chart legend to show or hide lines.</p>{bt_html}
<div class="foot">⚠️ Paper trading and research only, not financial advice. Backtests overstate real results.
Machine-readable data: <a href="results.json">results.json</a></div>
</div></body></html>"""
