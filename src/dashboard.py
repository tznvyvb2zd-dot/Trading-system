"""Builds docs/index.html: a mobile-friendly live dashboard for the UK (FTSE 350) system. Plotly via CDN."""
from __future__ import annotations

from html import escape

import pandas as pd
import plotly.graph_objects as go

BASE = {"Logistic regression": "#2a78d6", "LightGBM": "#eb6834", "Ensemble": "#1baf7a",
        "LR + earnings": "#6250d6", "LightGBM + earnings": "#e34948", "Ensemble + earnings": "#008300",
        "Momentum rule": "#eda100"}
FIXED = {"Equal-weight (all)": ("#e87ba4", "solid"), "FTSE 100 tracker": ("#52514e", "dash"),
         "FTSE 250 tracker": ("#9a988f", "dashdot")}
INK, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#e6e5e0", "#fcfcfb"
CONFIG = {"displayModeBar": False, "responsive": True}


def line_style(name):
    if name in FIXED:
        return FIXED[name]
    base, _, kind = name.partition(" · ")
    return BASE.get(base, "#888"), ("dot" if kind == "monthly" else "solid")


SHORT = {"Logistic regression": "LR", "LightGBM": "LGBM", "Ensemble": "Ens", "LR + earnings": "LR+E",
         "LightGBM + earnings": "LGBM+E", "Ensemble + earnings": "Ens+E", "Momentum rule": "Momentum"}


def legend_name(s):
    if " · " in s:
        base, _, kind = s.partition(" · ")
        return f"{SHORT.get(base, base)} {'daily' if 'daily' in kind else 'monthly'}"
    return {"Equal-weight (all)": "Equal-weight"}.get(s, s)


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


def _rank_badge(r, buffer):
    return f"🟢 {r}" if r <= 10 else (f"🟡 {r}" if r <= buffer else f"🔴 {r}")


def build_dashboard(results: dict, bt: dict, nav: pd.DataFrame) -> str:
    p, v, model = results["paper"], results["verdict"], results["model"]
    b100 = p["benchmarks"]["FTSE 100 tracker"]; b250 = p["benchmarks"]["FTSE 250 tracker"]
    ahead = p["value_gbp"] - b100["value_gbp"]
    buffer = results["buffer_rank"]

    # live chart
    if len(nav) >= 2:
        f = go.Figure()
        f.add_trace(go.Scatter(x=nav["date"], y=nav["portfolio_gbp"], name="My paper portfolio", mode="lines+markers",
                               line=dict(color=BASE.get(model, "#2a78d6"), width=2.5), marker=dict(size=6),
                               hovertemplate="£%{y:,.2f}"))
        for k in ("FTSE 100 tracker", "FTSE 250 tracker"):
            c, d = FIXED[k]
            f.add_trace(go.Scatter(x=nav["date"], y=nav[k], name=k, mode="lines+markers", marker=dict(size=5),
                                   line=dict(color=c, width=1.8, dash=d), hovertemplate="£%{y:,.2f}"))
        f.update_yaxes(tickprefix="£"); f.update_xaxes(type="date", tickformat="%d %b")
        live_chart = _style(f, "Paper portfolio vs FTSE trackers (live, £)", 300)
    else:
        live_chart = "<p class='muted'>The live chart appears after the second daily run.</p>"

    # today's activity
    td = results["today"]
    if td["trades"]:
        icon = {"BUY": "🟢 BUY", "SELL": "🔴 SELL", "TOP UP": "🔵 TOP UP", "TRIM": "🔵 TRIM"}
        today_html = _table(td["trades"], [("Stock", lambda t: t["ticker"].replace(".L", "")),
                                           ("Action", lambda t: icon.get(t["action"], t["action"])),
                                           ("Amount", lambda t: f"£{t['change_gbp']:+.2f}"), ("Cost", lambda t: f"£{t['cost_gbp']:.2f}"),
                                           ("Why", lambda t: t.get("why", ""))])
    else:
        today_html = (f"<p class='muted'>✅ Daily check: all holdings are still in the top {buffer}, so <b>no trades today</b>. "
                      "Nothing to do.</p>")
    da = td.get("drop_alerts") or []
    if da:
        today_html += ("<p class='muted' style='margin-top:10px'><b>⚠️ Sudden drops today</b> (8%+ in a day or 15%+ since bought). "
                       f"SELL only if the model also ranks it below {buffer}:</p>" + _table(da, [
                           ("Stock", lambda a: a["ticker"].replace(".L", "")), ("Today", lambda a: _pct(a["day"])),
                           ("Since bought", lambda a: _pct(a["since_entry"])), ("Model rank", lambda a: a["raw_rank"]),
                           ("Decision", lambda a: "🔴 SELL" if a["sell"] else "🟡 HOLD")]))

    holdings = _table(results["holdings"], [
        ("Stock", lambda h: h["ticker"].replace(".L", "")), ("Rank today", lambda h: _rank_badge(h["rank_today"], buffer)),
        ("Value", lambda h: f"£{h['value_gbp']:.2f}"), ("Since bought", lambda h: _pct(h["return_since_entry"])),
        ("Shares", lambda h: "-" if h["shares"] is None else f"{h['shares']:.4f}"), ("Sector", lambda h: h["sector"])])
    held = {h["ticker"] for h in results["holdings"]}
    watch = _table(results["watchlist"], [
        ("Rank", lambda w: w["rank"]), ("Stock", lambda w: w["ticker"].replace(".L", "")),
        ("Held?", lambda w: "✓" if w["ticker"] in held else ""), ("Score", lambda w: f"{w['score']:.3f}"),
        ("Sector", lambda w: w["sector"])])

    # backtest
    bt_html = ""
    if bt and bt.get("growth"):
        key = [f"{model} · daily check", f"{model} · monthly", "Equal-weight (all)", "FTSE 100 tracker", "FTSE 250 tracker"]
        g = go.Figure()
        for s, series in bt["growth"].items():
            c, d = line_style(s)
            g.add_trace(go.Scatter(x=list(series), y=list(series.values()), name=legend_name(s),
                                   visible=True if s in key else "legendonly",
                                   line=dict(color=c, width=2.5 if s == key[0] else 1.5, dash=d), hovertemplate="£%{y:,.0f}"))
        g.update_yaxes(type="log", tickprefix="£")
        bt_html += f"<div class='card'>{_style(g, '£100 invested · backtest, real FTSE 350', 480, two_col_legend=True)}</div>"

        dvm = bt.get("daily_vs_monthly") or []
        better = sum(r["sharpe_daily"] > r["sharpe_monthly"] for r in dvm)
        bt_html += ("<h2>🔁 Does the daily check help?</h2><div class='card'>" + _table(dvm, [
            ("Model", lambda r: r["model"]), ("Sharpe monthly", lambda r: f"{r['sharpe_monthly']:.2f}"),
            ("Sharpe daily check", lambda r: f"{r['sharpe_daily']:.2f}"),
            ("…without drop rule", lambda r: f"{r['sharpe_daily_no_drop_rule']:.2f}" if "sharpe_daily_no_drop_rule" in r else "next re-pick"),
            ("CAGR monthly → daily", lambda r: f"{r['cagr_monthly']:.1%} → {r['cagr_daily']:.1%}")]) +
            f"<p class='muted'>Daily check = sell a holding if its 10-day smoothed rank falls below {buffer} (max 2 swaps a day) and swap in the best replacement. "
            f"It improved Sharpe for {better} of {len(dvm)} strategies, after 0.5% stamp duty + spread on every buy.</p></div>")

        short = {"Logistic regression": "LR", "LightGBM": "LGBM", "Ensemble": "Ens", "LR + earnings": "LR+E",
                 "LightGBM + earnings": "LGBM+E", "Ensemble + earnings": "Ens+E", "Momentum rule": "Mom"}
        lab = lambda s: (short.get(s.split(" · ")[0], s.split(" · ")[0]) + (" d" if "daily" in s else " m")) if " · " in s else \
            {"Equal-weight (all)": "EqW", "FTSE 100 tracker": "F100", "FTSE 250 tracker": "F250"}.get(s, s)
        years = sorted({y for s in key for y in bt["yearly"].get(s, {})})
        z = [[bt["yearly"][s].get(y) for s in key] for y in years]
        hm = go.Figure(go.Heatmap(z=z, y=years, x=[lab(s) for s in key], zmid=0, zmin=-0.4, zmax=0.4,
                                  colorscale=[[0, "#e34948"], [0.5, "#f0efec"], [1, "#2a78d6"]],
                                  text=[[("" if c is None else f"{c:+.0%}") for c in r] for r in z], texttemplate="%{text}",
                                  hovertemplate="%{x}, %{y}: %{text}<extra></extra>", showscale=False))
        hm.update_layout(hovermode="closest"); hm.update_yaxes(autorange="reversed", type="category"); hm.update_xaxes(side="top")
        bt_html += f"<div class='card'>{_style(hm, 'Return each year (blue = gain, red = loss)', 120 + 26 * len(years))}</div>"

        rows = sorted(bt["table"].items(), key=lambda kv: -kv[1]["Sharpe"])[:12]
        bt_html += "<div class='card'>" + _table(rows, [
            ("Strategy", lambda r: r[0]), ("Sharpe", lambda r: f"{r[1]['Sharpe']:.2f}"), ("CAGR", lambda r: f"{r[1]['CAGR']:.1%}"),
            ("Worst fall", lambda r: f"{r[1]['Max drawdown']:.0%}"), ("£100 →", lambda r: f"£{r[1]['£100 grew to']:,.0f}")]) + "</div>"
        auc = ", ".join(f"{k} {val:.3f}" for k, val in (bt.get("auc") or {}).items())
        cov = bt.get("coverage") or {}
        cov_txt = f"{min(cov.values()):.0%}–{max(cov.values()):.0%}" if cov else "-"
        ec = bt.get("earnings_coverage")
        bt_html += (f"<p class='muted'>Backtest {escape(' → '.join(bt.get('period') or []))}, refreshed {escape(str(bt.get('generated')))}. "
                    f"AUC (0.50 = coin flip): {escape(auc)}. Price data covers {cov_txt} of FTSE 350 members"
                    + (f"; earnings data covers {ec:.0%} of stock-days" if ec is not None else "") + ".</p>")

    # --- "Your portfolio" calculator (runs in the browser)
    import json as _json
    tot = sum(h["value_gbp"] for h in results["holdings"]) or 1
    calc_data = {
        "holdings": [{"t": h["ticker"].replace(".L", ""), "s": h["sector"], "w": h["value_gbp"] / tot,
                      "px": h.get("price_gbp") or None, "r": h["rank_today"]} for h in results["holdings"]],
        "trades": [{"t": t["ticker"].replace(".L", ""), "a": t["action"], "c": t["change_gbp"]} for t in td["trades"]],
        "paper": p["value_gbp"], "buy_bps": results["costs"]["buy_bps"], "type": td["type"],
    }
    calc_json = _json.dumps(calc_data).replace("</", "<\\/")
    calc_html = f"""<div class='card'><label class='calc'>My amount to invest: <span class='pfx'>£</span>
<input id='amt' type='number' inputmode='decimal' min='1' step='1' value='100'></label>
<div id='calcSummary' class='muted'></div><div id='calcTable'></div>
<p class='muted'>Split = the system's current holdings, scaled to your amount. Buy with <b>value orders</b> on Trading 212
(e.g. "buy £24.85 of ROR"); shares are approximate (prices move). Each buy pays ~0.5% stamp duty + spread, already allowed for.
Your amount is saved on this device only.</p><div id='calcTrades'></div></div>
<script type='application/json' id='calcData'>{calc_json}</script>"""
    calc_js = """<script>(function(){
const D=JSON.parse(document.getElementById('calcData').textContent);
const inp=document.getElementById('amt');
const gbp=x=>'£'+x.toLocaleString('en-GB',{minimumFractionDigits:2,maximumFractionDigits:2});
const esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
try{const v=localStorage.getItem('tsAmount');if(v)inp.value=v;}catch(e){}
function draw(){
  let amt=parseFloat(inp.value); if(!(amt>0)) amt=0;
  try{localStorage.setItem('tsAmount',String(amt||''));}catch(e){}
  const k=1+D.buy_bps/1e4, budget=amt/k;
  let rows=D.holdings.map(h=>{const v=budget*h.w;return `<tr><td>${esc(h.t)}</td><td>${gbp(v)}</td><td>${h.px?(v/h.px).toFixed(4):'-'}</td><td>${h.px?gbp(h.px):'-'}</td><td>${esc(h.s)}</td></tr>`;}).join('');
  const small=D.holdings.some(h=>budget*h.w<1);
  document.getElementById('calcTable').innerHTML=`<div class='tw'><table class='tbl'><thead><tr><th>Stock</th><th>Invest</th><th>≈ Shares</th><th>Price</th><th>Sector</th></tr></thead><tbody>${rows}</tbody></table></div>`;
  document.getElementById('calcSummary').innerHTML=amt?`Invest <b>${gbp(budget)}</b> across ${D.holdings.length} stocks · about <b>${gbp(amt-budget)}</b> goes on stamp duty + spread`+(small?` · <b>⚠️ some amounts are under £1</b>, Trading 212's minimum order`:''):'Enter an amount above.';
  const f=D.paper?budget/D.paper:0;
  document.getElementById('calcTrades').innerHTML=D.trades.length&&amt?`<p class='muted' style='margin-top:10px'><b>Today's trades at your size (${esc(D.type)}):</b></p><div class='tw'><table class='tbl'><thead><tr><th>Stock</th><th>Action</th><th>Amount</th></tr></thead><tbody>${D.trades.map(t=>`<tr><td>${esc(t.t)}</td><td>${esc(t.a)}</td><td>${gbp(t.c*f)}</td></tr>`).join('')}</tbody></table></div>`:'';
}
inp.addEventListener('input',draw); draw();
})();</script>"""

    kpi = lambda label, value, sub="", cls="": (f"<div class='kpi {cls}'><div class='kl'>{escape(label)}</div>"
                                               f"<div class='kv'>{escape(value)}</div><div class='ks'>{escape(sub)}</div></div>")
    kpis = "".join([
        kpi("Paper portfolio", f"£{p['value_gbp']:,.2f}", f"{_pct(p['return_since_start'])} since {p['start_date']}"),
        kpi("FTSE 100 tracker", f"£{b100['value_gbp']:,.2f}", f"{_pct(b100['return_since_start'])} since start"),
        kpi("FTSE 250 tracker", f"£{b250['value_gbp']:,.2f}", f"{_pct(b250['return_since_start'])} since start"),
        kpi("vs FTSE 100", f"{'+' if ahead >= 0 else '-'}£{abs(ahead):,.2f}", "ahead" if ahead >= 0 else "behind",
            "pos" if ahead >= 0 else "neg"),
        kpi("Today", _pct(p.get("day_change")), td["type"]),
        kpi("Next full re-pick", f"{results['next_rebalance_in_trading_days']} trading days", f"last: {results['last_rebalance']}"),
    ])
    ok = v["beats_ftse100"] or v["beats_ftse250"]
    verdict = (f"<div class='note {'pos' if ok else 'neg'}'>Backtest verdict: <b>{escape(model)}</b> with daily check "
               f"{'beats' if ok else 'does <b>not</b> beat'} the FTSE trackers on risk-adjusted return "
               f"(Sharpe {v['model_sharpe']:.2f} vs FTSE 100 {v['ftse100_sharpe']:.2f}, FTSE 250 {v['ftse250_sharpe']:.2f}). "
               "Treat this as a live experiment.</div>")

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
.calc{display:flex;align-items:center;gap:8px;font-weight:600;font-size:15px;margin:4px 0 8px;flex-wrap:wrap}
.calc .pfx{font-size:20px}
.calc input{font-size:20px;font-weight:650;width:9em;padding:8px 10px;border:1px solid #c9c7bf;border-radius:10px;background:#fff}
"""
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta name="robots" content="noindex">
<title>UK Trading System · {escape(results['market_date'])}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>{css}</style></head><body><div class="wrap">
<h1>🇬🇧 UK Trading System · FTSE 350</h1>
<div class="muted">Prices to {escape(results['market_date'])} · updated {escape(results['generated_at_utc'])} UTC ·
£100 paper portfolio · full re-pick monthly + daily check (sell if 10-day smoothed rank &gt; {buffer}, max 2 swaps/day; sudden drops sold if the model agrees) · 📲 phone alerts via ntfy · costs include 0.5% stamp duty on buys</div>
<div class="kpis">{kpis}</div>{verdict}
<h2>📅 Today ({escape(td['type'])})</h2><div class="card">{today_html}</div>
<h2>🧮 Your portfolio: how much to buy</h2>{calc_html}
<h2>📈 Live paper trading</h2><div class="card">{live_chart}</div>
<div class="grid2"><div><h2>💼 Holdings (🟢 top 10 · 🟡 top {buffer} · 🔴 sell)</h2><div class="card">{holdings}</div></div>
<div><h2>👀 Today's top 15 (smoothed)</h2><div class="card">{watch}</div></div></div>
<h2>📊 Backtest (no hindsight)</h2><p class="muted">Tip: tap names in a chart legend to show or hide lines. Solid = daily check, dotted = monthly only. LR = logistic regression, LGBM = LightGBM, Ens = ensemble, +E = with earnings.</p>{bt_html}
<div class="foot">⚠️ Paper trading and research only, not financial advice. Backtests overstate real results.
Machine-readable data: <a href="results.json">results.json</a></div>
</div>{calc_js}</body></html>"""
