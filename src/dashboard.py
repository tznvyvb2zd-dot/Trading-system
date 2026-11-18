"""Builds docs/index.html: dark, app-style (tabbed) live dashboard for the UK (FTSE 350) system. Plotly via CDN."""
from __future__ import annotations

import json
from html import escape

import pandas as pd
import plotly.graph_objects as go

REPO = "https://github.com/tznvyvb2zd-dot/Trading-system"
EDIT_URL = f"{REPO}/edit/main/docs/my_portfolio.json"

# dark-mode chart colours (validated categorical steps for dark surfaces)
BASE = {"Logistic regression": "#3987e5", "LightGBM": "#d95926", "Ensemble": "#199e70",
        "LR + earnings": "#9085e9", "LightGBM + earnings": "#e66767", "Ensemble + earnings": "#0ca30c",
        "Momentum rule": "#c98500"}
FIXED = {"Equal-weight (all)": ("#d55181", "solid"), "FTSE 100 tracker": ("#a3acbd", "dash"),
         "FTSE 250 tracker": ("#6b7588", "dashdot")}
BG, CARD, LINE, TEXT, MUTED = "#0B0F17", "#121826", "#1F2937", "#E6EAF2", "#8B95A7"
CONFIG = {"displayModeBar": False, "responsive": True}
SHORT = {"Logistic regression": "LR", "LightGBM": "LGBM", "Ensemble": "Ens", "LR + earnings": "LR+E",
         "LightGBM + earnings": "LGBM+E", "Ensemble + earnings": "Ens+E", "Momentum rule": "Momentum"}


def line_style(name):
    if name in FIXED:
        return FIXED[name]
    base, _, kind = name.partition(" · ")
    return BASE.get(base, "#888"), ("dot" if kind == "monthly" else "solid")


def legend_name(s):
    if " · " in s:
        base, _, kind = s.partition(" · ")
        return f"{SHORT.get(base, base)} {'daily' if 'daily' in kind else 'monthly'}"
    return {"Equal-weight (all)": "Equal-weight"}.get(s, s)


def _style(fig, title, h=320, legend_cols=False):
    fig.update_layout(title=dict(text=title, x=0, font=dict(size=14, color=TEXT)), height=h, template="plotly_dark",
                      font=dict(family="Inter,system-ui,-apple-system,Segoe UI,sans-serif", color=MUTED, size=11.5),
                      margin=dict(l=46, r=10, t=44, b=28), hovermode="x unified", paper_bgcolor=CARD, plot_bgcolor=CARD,
                      hoverlabel=dict(bgcolor="#1B2333", bordercolor=LINE, font=dict(color=TEXT)),
                      legend=dict(orientation="h", y=-0.16, font=dict(size=10.5, color=MUTED)))
    fig.update_xaxes(gridcolor=LINE, linecolor=LINE, zeroline=False)
    fig.update_yaxes(gridcolor=LINE, linecolor=LINE, zeroline=False)
    if legend_cols:
        fig.update_layout(legend=dict(orientation="h", y=-0.1, yanchor="top", entrywidth=0.5, entrywidthmode="fraction"))
    fig.update_layout(autosize=True)
    return fig.to_html(full_html=False, include_plotlyjs=False, config=CONFIG,
                       default_width="100%", default_height=f"{h}px")


def _pct(x, sign=True):
    return "–" if x is None else (f"{x:+.2%}" if sign else f"{x:.1%}")


def _cls(x):
    return "" if x is None else ("pos" if x >= 0 else "neg")


def _table(rows, cols):
    head = "".join(f"<th>{escape(c)}</th>" for c, _ in cols)
    body = "".join("<tr>" + "".join(f"<td>{fn(r)}</td>" for _, fn in cols) + "</tr>" for r in rows)
    return f"<div class='tw'><table class='tbl'><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def _badge(r, buffer):
    cls = "b-top" if r <= 10 else ("b-mid" if r <= buffer else "b-out")
    return f"<span class='badge {cls}'>#{r}</span>"


def build_dashboard(results: dict, bt: dict, nav: pd.DataFrame) -> str:
    v, model, buffer = results["verdict"], results["model"], results["buffer_rank"]
    td = results["today"]
    e = lambda x: escape(str(x))
    nm = lambda t: e(t.replace(".L", ""))

    # ---------------- data for the in-browser tracker
    tot = sum(h["value_gbp"] for h in results["holdings"]) or 1
    cols = ["date", "portfolio_gbp", "FTSE 100 tracker", "FTSE 250 tracker"]
    data = {
        "nav": nav[cols].round(4).values.tolist() if len(nav) else [],
        "paper": results["paper"]["value_gbp"],
        "holdings": [{"t": h["ticker"].replace(".L", ""), "s": h["sector"], "w": h["value_gbp"] / tot,
                      "px": h.get("price_gbp"), "r": h["rank_today"]} for h in results["holdings"]],
        "trades": [{"t": t["ticker"].replace(".L", ""), "a": t["action"], "c": t["change_gbp"], "why": t.get("why", "")}
                   for t in td["trades"]],
        "default": results.get("my_portfolio") or {"amount_gbp": 100, "start_date": results["paper"]["start_date"]},
    }
    data_json = json.dumps(data).replace("</", "<\\/")

    # ---------------- Today tab (static parts)
    da = td.get("drop_alerts") or []
    drops = ""
    if da:
        drops = ("<div class='card'><div class='card-h'>⚠️ Sudden drops today</div><p class='muted small'>Down 8%+ today or 15%+ "
                 f"since bought. Sold only if the model also ranks it below #{buffer}.</p>" + _table(da, [
                     ("Stock", lambda a: f"<b>{nm(a['ticker'])}</b>"),
                     ("Today", lambda a: f"<span class='{_cls(a['day'])}'>{_pct(a['day'])}</span>"),
                     ("Since bought", lambda a: f"<span class='{_cls(a['since_entry'])}'>{_pct(a['since_entry'])}</span>"),
                     ("Rank", lambda a: f"#{a['raw_rank']}"),
                     ("Decision", lambda a: "<span class='pill neg'>SELL</span>" if a["sell"] else "<span class='pill warn'>HOLD</span>")])
                 + "</div>")
    ok = v["beats_ftse100"] and v["beats_ftse250"]
    verdict = (f"<div class='note {'pos' if ok else 'neg'}'><b>Backtest verdict:</b> {e(model)} with daily check "
               f"{'beats both' if ok else 'does <b>not</b> beat both'} FTSE trackers on risk-adjusted return (Sharpe "
               f"{v['model_sharpe']:.2f} vs FTSE 100 {v['ftse100_sharpe']:.2f} · FTSE 250 {v['ftse250_sharpe']:.2f}). "
               "Treat this as a live experiment, not advice.</div>")

    # ---------------- Holdings tab
    holdings = _table(results["holdings"], [
        ("Stock", lambda h: f"<b>{nm(h['ticker'])}</b>"), ("Rank", lambda h: _badge(h["rank_today"], buffer)),
        ("Since bought", lambda h: f"<span class='{_cls(h['return_since_entry'])}'>{_pct(h['return_since_entry'])}</span>"),
        ("Paper £", lambda h: f"£{h['value_gbp']:.2f}"), ("Sector", lambda h: f"<span class='muted'>{e(h['sector'])}</span>")])
    held = {h["ticker"] for h in results["holdings"]}
    watch = _table(results["watchlist"], [
        ("#", lambda w: w["rank"]), ("Stock", lambda w: f"<b>{nm(w['ticker'])}</b>"),
        ("", lambda w: "<span class='pill acc'>held</span>" if w["ticker"] in held else ""),
        ("Score", lambda w: f"{w['score']:.3f}"), ("Sector", lambda w: f"<span class='muted'>{e(w['sector'])}</span>")])

    # ---------------- Backtest tab
    bt_html = ""
    if bt and bt.get("growth"):
        key = [f"{model} · daily check", f"{model} · monthly", "Equal-weight (all)", "FTSE 100 tracker", "FTSE 250 tracker"]
        g = go.Figure()
        for s, series in bt["growth"].items():
            c, d = line_style(s)
            g.add_trace(go.Scatter(x=list(series), y=list(series.values()), name=legend_name(s),
                                   visible=True if s in key else "legendonly",
                                   line=dict(color=c, width=2.6 if s == key[0] else 1.5, dash=d), hovertemplate="£%{y:,.0f}"))
        g.update_yaxes(type="log", tickprefix="£", dtick=0.30103, minor=dict(showgrid=False))      # gridlines at each doubling
        bt_html += f"<div class='card plot'>{_style(g, '£100 invested · backtest on the real FTSE 350', 470, legend_cols=True)}</div>"
        dvm = bt.get("daily_vs_monthly") or []
        bt_html += ("<div class='card'><div class='card-h'>🔁 Does the daily check help? (Sharpe)</div>" + _table(dvm, [
            ("Model", lambda r: f"<b>{e(SHORT.get(r['model'], r['model']))}</b>"),
            ("Monthly", lambda r: f"{r['sharpe_monthly']:.2f}"), ("Daily check", lambda r: f"{r['sharpe_daily']:.2f}"),
            ("No drop rule", lambda r: f"{r['sharpe_daily_no_drop_rule']:.2f}" if "sharpe_daily_no_drop_rule" in r else "–")]) +
            "<p class='muted small'>Sharpe = return per unit of risk (higher is better), after 0.5% stamp duty + spread on every buy.</p></div>")
        lab = lambda s: (SHORT.get(s.split(" · ")[0], s) + (" d" if "daily" in s else " m")) if " · " in s else \
            {"Equal-weight (all)": "EqW", "FTSE 100 tracker": "F100", "FTSE 250 tracker": "F250"}.get(s, s)
        years = sorted({y for s in key for y in bt["yearly"].get(s, {})})
        z = [[bt["yearly"][s].get(y) for s in key] for y in years]
        hm = go.Figure(go.Heatmap(z=z, y=years, x=[lab(s) for s in key], zmid=0, zmin=-0.4, zmax=0.4,
                                  colorscale=[[0, "#e66767"], [0.5, "#1F2937"], [1, "#3987e5"]],
                                  text=[[("" if c is None else f"{c:+.0%}") for c in r] for r in z], texttemplate="%{text}",
                                  textfont=dict(color=TEXT), hovertemplate="%{x}, %{y}: %{text}<extra></extra>", showscale=False))
        hm.update_layout(hovermode="closest"); hm.update_yaxes(autorange="reversed", type="category"); hm.update_xaxes(side="top")
        bt_html += f"<div class='card plot'>{_style(hm, 'Return each year (blue = gain · red = loss)', 110 + 26 * len(years))}</div>"
        rows = sorted(bt["table"].items(), key=lambda kv: -kv[1]["Sharpe"])[:12]
        bt_html += "<div class='card'><div class='card-h'>🏁 Top strategies</div>" + _table(rows, [
            ("Strategy", lambda r: f"<b>{e(legend_name(r[0]))}</b>"), ("Sharpe", lambda r: f"{r[1]['Sharpe']:.2f}"),
            ("CAGR", lambda r: f"<span class='{_cls(r[1]['CAGR'])}'>{r[1]['CAGR']:.1%}</span>"),
            ("Worst fall", lambda r: f"<span class='neg'>{r[1]['Max drawdown']:.0%}</span>"),
            ("£100 →", lambda r: f"£{r[1]['£100 grew to']:,.0f}")]) + "</div>"
        auc = ", ".join(f"{SHORT.get(k, k)} {val:.3f}" for k, val in (bt.get("auc") or {}).items())
        cov = bt.get("coverage") or {}
        cov_txt = f"{min(cov.values()):.0%}–{max(cov.values()):.0%}" if cov else "–"
        bt_html += (f"<p class='muted small'>Backtest {e(' → '.join(bt.get('period') or []))}, refreshed {e(bt.get('generated'))}. "
                    f"Prediction accuracy (AUC, 0.50 = coin flip): {e(auc)}. Price data covers {cov_txt} of FTSE 350 members. "
                    "Solid = daily check, dotted = monthly only. Tap legend names to show/hide lines.</p>")

    js = JS.replace("__BUF__", str(buffer))
    return f"""<!DOCTYPE html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover"><meta name="robots" content="noindex">
<meta name="theme-color" content="{BG}"><meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<title>Trading System · {e(results['market_date'])}</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script><style>{CSS}</style></head><body>
<header class="top"><div><div class="brand">🇬🇧 Trading System</div>
<div class="sub">FTSE 350 · prices to {e(results['market_date'])} · updated {e(results['generated_at_utc'])} UTC</div></div>
<span class="pill {'acc' if td['type'] == 'daily check' else 'warn'}">{e(td['type'])}</span></header>
<main>
<section id="tab-me" class="tab active">
  <div class="hero card glow">
    <div class="hero-l">My portfolio</div>
    <div id="myValue" class="hero-v">£–</div>
    <div id="myChange" class="hero-c"></div>
    <div id="mySince" class="muted small"></div>
    <div class="chips"><div class="chip"><span>Same £ in FTSE 100</span><b id="vs100">–</b></div>
      <div class="chip"><span>Same £ in FTSE 250</span><b id="vs250">–</b></div>
      <div class="chip"><span>vs FTSE 100</span><b id="ahead">–</b></div></div>
  </div>
  <div class="card plot"><div id="myChart"></div><p id="myChartNote" class="muted small pad"></p></div>
  <div class="card"><div class="card-h">💼 Your holdings at your size</div><div id="mySplit"></div>
    <p class="muted small">Your current value split across the system's 10 holdings. Buying from scratch? Use <b>value orders</b>
    on Trading 212 (e.g. "buy £8.50 of ROR"); about 0.6% goes on stamp duty + spread.</p></div>
  <a class="btn" href="{EDIT_URL}" target="_blank" rel="noopener">✏️ Change my amount or start date</a>
  <p class="muted small center">Opens GitHub: edit the numbers, tap <b>Commit changes</b>, and the dashboard updates in about a minute.
  The tracker assumes you follow every trade.</p>
</section>

<section id="tab-today" class="tab">
  <div class="card"><div class="card-h">📅 Today: {e(td['type'])}</div><div id="todayTrades"></div>
    <div class="kv-row"><div><span class="muted">Next full re-pick</span><b>{results['next_rebalance_in_trading_days']} trading days</b></div>
    <div><span class="muted">Last re-pick</span><b>{e(results['last_rebalance'])}</b></div>
    <div><span class="muted">Model</span><b>{e(model)}</b></div></div></div>
  {drops}{verdict}
  <p class="muted small">Daily check: a holding is sold if its 10-day smoothed rank falls below #{buffer} (max 2 swaps a day), or if it
  drops 8%+ in a day / 15%+ since bought <i>and</i> the model ranks it below #{buffer}. Alerts go to your phone via ntfy.</p>
</section>

<section id="tab-hold" class="tab">
  <div class="card"><div class="card-h">💼 Holdings <span class="muted small">· green top 10 · amber top {buffer} · red = sell zone</span></div>{holdings}</div>
  <div class="card"><div class="card-h">👀 Today's top 15 <span class="muted small">· 10-day smoothed</span></div>{watch}</div>
</section>

<section id="tab-bt" class="tab">{verdict}{bt_html}
  <p class="muted small">⚠️ Paper trading and research only, not financial advice. Backtests overstate real results.
  Data: <a href="results.json">results.json</a> · <a href="{REPO}">GitHub</a></p>
</section>
</main>
<nav class="tabbar" role="tablist">
  <button class="active" data-tab="tab-me" role="tab"><span>💷</span>Portfolio</button>
  <button data-tab="tab-today" role="tab"><span>📅</span>Today</button>
  <button data-tab="tab-hold" role="tab"><span>📊</span>Holdings</button>
  <button data-tab="tab-bt" role="tab"><span>🧪</span>Backtest</button>
</nav>
<script type="application/json" id="D">{data_json}</script>
<script>{js}</script></body></html>"""


CSS = """
:root{--bg:#0B0F17;--card:#121826;--card2:#171F2F;--line:#1F2937;--text:#E6EAF2;--muted:#8B95A7;--acc:#4F8CFF;
--pos:#2FD27A;--neg:#FF5C6C;--warn:#F5B941;color-scheme:dark}
*{box-sizing:border-box}html,body{margin:0;background:var(--bg);color:var(--text);
font-family:Inter,system-ui,-apple-system,'Segoe UI',sans-serif;-webkit-font-smoothing:antialiased}
a{color:var(--acc)}b{font-weight:650}
.top{position:sticky;top:0;z-index:5;display:flex;justify-content:space-between;align-items:center;gap:10px;
padding:calc(12px + env(safe-area-inset-top)) 16px 12px;background:rgba(11,15,23,.88);backdrop-filter:blur(12px);
-webkit-backdrop-filter:blur(12px);border-bottom:1px solid var(--line)}
.brand{font-weight:700;font-size:17px;letter-spacing:.2px}.sub{color:var(--muted);font-size:11.5px;margin-top:2px}
main{max-width:980px;margin:0 auto;padding:14px 14px 100px}
.tab{display:none}.tab.active{display:block;animation:fade .18s ease}@keyframes fade{from{opacity:0;transform:translateY(4px)}to{opacity:1}}
.card{background:var(--card);border:1px solid var(--line);border-radius:18px;padding:14px;margin-bottom:12px;min-width:0}
.card.plot{padding:6px 6px 2px;overflow:hidden}.card-h{font-weight:650;font-size:14.5px;margin:2px 0 10px}
.glow{background:radial-gradient(120% 140% at 0% 0%,rgba(79,140,255,.22),transparent 55%),var(--card);
box-shadow:0 10px 30px rgba(0,0,0,.35),inset 0 0 0 1px rgba(79,140,255,.10)}
.hero{padding:18px 16px}.hero-l{color:var(--muted);font-size:13px}
.hero-v{font-size:42px;font-weight:750;letter-spacing:-1px;margin:2px 0;font-variant-numeric:tabular-nums}
.hero-c{font-size:15px;font-weight:650;margin-bottom:4px}
.chips{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:14px}
.chip{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:9px 10px;min-width:0}
.chip span{display:block;color:var(--muted);font-size:10.5px;line-height:1.3}.chip b{font-size:14px;white-space:nowrap}
.pos{color:var(--pos)}.neg{color:var(--neg)}.muted{color:var(--muted)}.small{font-size:12px;line-height:1.5}
.center{text-align:center}.pad{padding:0 8px}
.pill{display:inline-block;font-size:11.5px;font-weight:650;padding:4px 10px;border-radius:999px;border:1px solid var(--line);white-space:nowrap}
.pill.acc{color:var(--acc);background:rgba(79,140,255,.12);border-color:rgba(79,140,255,.35)}
.pill.warn{color:var(--warn);background:rgba(245,185,65,.12);border-color:rgba(245,185,65,.35)}
.pill.neg{color:var(--neg);background:rgba(255,92,108,.12);border-color:rgba(255,92,108,.35)}
.pill.pos{color:var(--pos);background:rgba(47,210,122,.12);border-color:rgba(47,210,122,.35)}
.badge{display:inline-block;min-width:40px;text-align:center;font-size:12px;font-weight:700;padding:3px 8px;border-radius:8px}
.b-top{color:var(--pos);background:rgba(47,210,122,.12)}.b-mid{color:var(--warn);background:rgba(245,185,65,.12)}
.b-out{color:var(--neg);background:rgba(255,92,108,.12)}
.tw{overflow-x:auto}.tbl{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
.tbl th,.tbl td{padding:9px 8px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}
.tbl tr:last-child td{border-bottom:0}
.tbl th{color:var(--muted);font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.4px}
.tbl th:first-child,.tbl td:first-child{text-align:left}
.note{border-radius:14px;padding:12px 14px;font-size:13px;margin-bottom:12px;border:1px solid;line-height:1.5}
.note.pos{background:rgba(47,210,122,.08);border-color:rgba(47,210,122,.3)}
.note.neg{background:rgba(255,92,108,.07);border-color:rgba(255,92,108,.28)}
.kv-row{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:12px}
.kv-row div{background:var(--card2);border:1px solid var(--line);border-radius:12px;padding:9px 10px;min-width:0}
.kv-row span{display:block;font-size:11px}.kv-row b{font-size:13px;overflow-wrap:anywhere}
.btn{display:block;text-align:center;text-decoration:none;color:#fff;font-weight:650;background:var(--acc);
border-radius:14px;padding:13px;margin:6px 0 8px;box-shadow:0 8px 20px rgba(79,140,255,.25)}
.tabbar{position:fixed;left:0;right:0;bottom:0;z-index:10;display:grid;grid-template-columns:repeat(4,1fr);
background:rgba(14,19,29,.94);backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);border-top:1px solid var(--line);
padding:6px 6px calc(6px + env(safe-area-inset-bottom))}
.tabbar button{background:none;border:0;color:var(--muted);font:600 11px Inter,system-ui,sans-serif;padding:6px 0;border-radius:12px;cursor:pointer}
.tabbar button span{display:block;font-size:19px;margin-bottom:2px;filter:grayscale(1);opacity:.65}
.tabbar button.active{color:var(--text);background:rgba(79,140,255,.13)}.tabbar button.active span{filter:none;opacity:1}
@media(min-width:900px){.tabbar{left:50%;right:auto;transform:translateX(-50%);width:460px;bottom:16px;border:1px solid var(--line);
border-radius:18px;box-shadow:0 10px 30px rgba(0,0,0,.4)}}
"""

JS = r"""
(function(){
const D = JSON.parse(document.getElementById('D').textContent);
const C = {acc:'#4F8CFF', f100:'#a3acbd', f250:'#6b7588', card:'#121826', line:'#1F2937', text:'#E6EAF2', muted:'#8B95A7'};
const gbp = x => (x<0?'-':'') + '£' + Math.abs(x).toLocaleString('en-GB',{minimumFractionDigits:2,maximumFractionDigits:2});
const pct = x => (x>=0?'+':'') + (x*100).toFixed(2) + '%';
const esc = s => String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
const cls = x => x>=0 ? 'pos' : 'neg';
const fmtDate = s => { const d = new Date(s + 'T00:00:00'); return isNaN(d) ? s : d.toLocaleDateString('en-GB',{day:'numeric',month:'short',year:'numeric'}); };

document.querySelectorAll('.tabbar button').forEach(b => b.addEventListener('click', () => {
  document.querySelectorAll('.tabbar button').forEach(x => x.classList.toggle('active', x === b));
  document.querySelectorAll('.tab').forEach(t => t.classList.toggle('active', t.id === b.dataset.tab));
  document.querySelectorAll('#' + b.dataset.tab + ' .js-plotly-plot').forEach(p => window.Plotly && Plotly.Plots.resize(p));
  window.scrollTo({top:0});
}));

function render(cfg){
  const amt = +cfg.amount_gbp || 0, start = String(cfg.start_date || '');
  const rows = D.nav.filter(r => r[0] >= start);
  let series = [];
  if (rows.length) {
    const b = rows[0];
    series = rows.map(r => ({d:r[0], me:amt*r[1]/b[1], f100:amt*r[2]/b[2], f250:amt*r[3]/b[3]}));
  }
  const last = series.length ? series[series.length-1] : {me:amt, f100:amt, f250:amt};
  const ch = last.me - amt, r = amt ? ch/amt : 0;
  document.getElementById('myValue').textContent = gbp(last.me);
  document.getElementById('myChange').innerHTML = `<span class="${cls(ch)}">${ch>=0?'▲':'▼'} ${gbp(Math.abs(ch))} · ${pct(r)}</span>`;
  document.getElementById('mySince').textContent = `Started with ${gbp(amt)} on ${fmtDate(start)}` +
    (series.length ? '' : ' · tracking begins at the first market close on/after this date');
  document.getElementById('vs100').textContent = gbp(last.f100);
  document.getElementById('vs250').textContent = gbp(last.f250);
  const ah = last.me - last.f100, a = document.getElementById('ahead');
  a.textContent = (ah>=0?'+':'-') + gbp(Math.abs(ah)); a.className = cls(ah);

  const el = document.getElementById('myChart'), note = document.getElementById('myChartNote');
  if (window.Plotly && series.length >= 2) {
    const x = series.map(s => s.d);
    el.style.height = '300px';
    Plotly.newPlot(el, [
      {x, y:series.map(s=>s.me), name:'My portfolio', mode:'lines+markers', line:{color:C.acc, width:3, shape:'spline'},
       marker:{size:6}, hovertemplate:'£%{y:,.2f}'},
      {x, y:series.map(s=>s.f100), name:'FTSE 100 tracker', mode:'lines', line:{color:C.f100, width:1.6, dash:'dash'}, hovertemplate:'£%{y:,.2f}'},
      {x, y:series.map(s=>s.f250), name:'FTSE 250 tracker', mode:'lines', line:{color:C.f250, width:1.6, dash:'dashdot'}, hovertemplate:'£%{y:,.2f}'}
    ], {height:300, paper_bgcolor:C.card, plot_bgcolor:C.card, margin:{l:48,r:10,t:36,b:28}, hovermode:'x unified',
        title:{text:'My portfolio vs the same £ in FTSE trackers', x:0, font:{size:14, color:C.text}},
        font:{family:'Inter,system-ui,sans-serif', color:C.muted, size:11.5}, legend:{orientation:'h', y:-0.2},
        hoverlabel:{bgcolor:'#1B2333', bordercolor:C.line, font:{color:C.text}},
        xaxis:{type:'date', tickformat:'%d %b', gridcolor:C.line, linecolor:C.line, zeroline:false},
        yaxis:{tickprefix:'£', gridcolor:C.line, linecolor:C.line, zeroline:false, autorange:true, rangemode:'normal'}},
      {displayModeBar:false, responsive:true});
    note.textContent = '';
  } else {
    el.style.height = '0px';
    note.innerHTML = '📈 <b>Your chart starts soon.</b> It appears after two market closes from your start date (the system updates every weekday evening).';
  }

  const value = last.me;
  document.getElementById('mySplit').innerHTML = D.holdings.length ? `<div class='tw'><table class='tbl'><thead><tr><th>Stock</th><th>Your £</th><th>≈ Shares</th><th>Rank</th></tr></thead><tbody>` +
    D.holdings.map(h => { const v = value*h.w; return `<tr><td><b>${esc(h.t)}</b></td><td>${gbp(v)}</td><td>${h.px ? (v/h.px).toFixed(4) : '–'}</td><td>#${h.r}</td></tr>`; }).join('') +
    `</tbody></table></div>` : `<p class="muted">No holdings yet: the first picks arrive at tonight's run.</p>`;

  const k = D.paper ? value / D.paper : 0;
  document.getElementById('todayTrades').innerHTML = D.trades.length
    ? `<div class='tw'><table class='tbl'><thead><tr><th>Stock</th><th>Action</th><th>Your £</th><th>Why</th></tr></thead><tbody>` +
      D.trades.map(t => `<tr><td><b>${esc(t.t)}</b></td><td><span class="pill ${t.a==='SELL'?'neg':'pos'}">${esc(t.a)}</span></td><td>${gbp(Math.abs(t.c)*k)}</td><td class="muted">${esc(t.why)}</td></tr>`).join('') +
      `</tbody></table></div>`
    : `<p class="muted">✅ No trades today: every holding is still in the model's top __BUF__. Nothing to do.</p>`;
}

render(D.default);
fetch('my_portfolio.json?ts=' + Date.now(), {cache:'no-store'}).then(r => r.ok ? r.json() : null)
  .then(j => { if (j && j.amount_gbp) render(j); }).catch(() => {});
})();
"""
