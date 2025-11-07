
# signal_bot_news_full.py
"""
Integrated Telegram trading signal bot:
- Technicals: EMA/RSi/ATR/A-D/Volume
- Whale/spoofing detection via orderbook snapshots
- Elliott Wave heuristic (ZigZag) detection for strong wave-3
- Auto TP/SL via ATR factors
- News monitoring: NewsAPI (en/ar + domains=bloomberg.com) + GDELT GKG
- News -> asset impact heuristic and sentiment scoring
- Sends signals/alerts to Telegram, stores history in SQLite
Run:
  export TG_BOT_TOKEN="..."
  export TG_CHAT_ID="..."
  export NEWSAPI_KEY="..."
  python signal_bot_news_full.py
"""
import os, asyncio, logging, json, math, sqlite3, time, html
from datetime import datetime, timedelta
import aiohttp
import pandas as pd
import ccxt.async_support as ccxt
import pandas_ta as ta
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder, CommandHandler, ContextTypes

# ----- Logging -----
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("signal-bot-news")

# ----- DB -----
DB = "signals_news.db"
def init_db():
    conn = sqlite3.connect(DB)
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS signals(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, pair TEXT, timeframe TEXT,
        signal TEXT, entry REAL, stop REAL, tp REAL, confidence REAL, reason TEXT)""")
    cur.execute("""CREATE TABLE IF NOT EXISTS events(
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT, source TEXT, title TEXT, url TEXT, language TEXT, 
        impact TEXT, assets TEXT, sentiment REAL)""")
    conn.commit(); conn.close()

def save_signal(pair,tf,sig,entry,stop,tp,conf,reason):
    conn = sqlite3.connect(DB); cur = conn.cursor()
    cur.execute("INSERT INTO signals(ts,pair,timeframe,signal,entry,stop,tp,confidence,reason) VALUES (?,?,?,?,?,?,?,?,?)",
                (datetime.utcnow().isoformat(),pair,tf,sig,entry,stop,tp,conf,reason))
    conn.commit(); conn.close()

def save_event(src,title,url,lang,impact,assets,sentiment):
    conn = sqlite3.connect(DB); cur = conn.cursor()
    cur.execute("INSERT INTO events(ts,source,title,url,language,impact,assets,sentiment) VALUES (?,?,?,?,?,?,?)",
                (datetime.utcnow().isoformat(),src,title,url,lang,impact,",".join(assets),sentiment))
    conn.commit(); conn.close()

# ----- Load config -----
CONFIG_PATH = os.getenv("RULES_CONFIG","rules_config.json")
def load_config():
    with open(CONFIG_PATH,"r",encoding="utf-8") as f:
        return json.load(f)

# ----- Helpers: Telegram -----
async def send_msg(app, chat_id, text):
    try:
        await app.bot.send_message(chat_id=chat_id, text=text, parse_mode=ParseMode.MARKDOWN)
    except Exception as e:
        logger.warning("Telegram send error: %s", e)

# ----- Market data via CCXT -----
async def fetch_ohlcv(exchange_id, symbol, timeframe, limit=500):
    ex = getattr(ccxt, exchange_id)()
    await ex.load_markets()
    data = await ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    await ex.close()
    df = pd.DataFrame(data, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    return df

async def fetch_orderbook(exchange_id, symbol, depth=100):
    ex = getattr(ccxt, exchange_id)()
    await ex.load_markets()
    ob = await ex.fetch_order_book(symbol, limit=depth)
    await ex.close()
    return ob

# ----- Indicators -----
def add_indicators(df):
    df = df.copy()
    df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
    df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
    df["rsi_14"] = ta.rsi(df["close"], length=14)
    df["atr_14"] = ta.atr(df["high"], df["low"], df["close"], length=14)
    df["ad"] = ta.ad(df["high"], df["low"], df["close"], df["volume"])
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    return df

# ----- ZigZag simple (swings) -----
def zigzag(df, pct=0.02):
    prices = df["close"].values
    times = df.index
    pivots=[]
    last_pivot = prices[0]
    direction = None
    for i in range(1,len(prices)):
        change = (prices[i] - last_pivot) / last_pivot
        if direction is None:
            if abs(change)>=pct:
                direction = "up" if change>0 else "down"
                last_pivot = prices[i]; pivots.append((times[i], prices[i], direction))
        else:
            if direction=="up":
                if prices[i] > last_pivot:
                    last_pivot = prices[i]
                elif (prices[i]-last_pivot)/last_pivot <= -pct:
                    direction="down"; last_pivot = prices[i]; pivots.append((times[i], prices[i], "down"))
            else:
                if prices[i] < last_pivot:
                    last_pivot = prices[i]
                elif (prices[i]-last_pivot)/last_pivot >= pct:
                    direction="up"; last_pivot = prices[i]; pivots.append((times[i], prices[i], "up"))
    return pivots

def detect_wave3(pivots):
    # Heuristic: require at least two upward swings and third > 1.6× first
    ups = [p for p in pivots if p[2]=="up"]
    if len(ups) < 2: return {"detected":False}
    wave1 = ups[-2][1]; wave3 = ups[-1][1]
    ratio = (wave3 / wave1) if wave1 else 0
    if ratio > 1.6:
        conf = min(0.95, 0.2 + (ratio-1.6))  # coarse
        return {"detected":True, "ratio":ratio, "confidence":conf}
    return {"detected":False}

# ----- Whale / Spoofing detector -----
async def detect_spoofing(exchange_id, symbol, depth=50, threshold_usd=100000):
    ob = await fetch_orderbook(exchange_id, symbol, depth=depth)
    bids = ob.get("bids", []); asks = ob.get("asks", [])
    top_bids = sum([p*q for p,q in bids[:5]]) if bids else 0
    top_asks = sum([p*q for p,q in asks[:5]]) if asks else 0
    if top_bids >= threshold_usd and top_bids > top_asks * 2:
        return {"side":"buy","value":top_bids}
    if top_asks >= threshold_usd and top_asks > top_bids * 2:
        return {"side":"sell","value":top_asks}
    return None

# ----- Evaluate technical rules -----
def evaluate_technical(df, pair):
    latest = df.iloc[-1]; prev = df.iloc[-2]
    signals=[]
    # EMA crossover
    if prev["ema_12"] <= prev["ema_26"] and latest["ema_12"] > latest["ema_26"]:
        signals.append(("BUY","EMA12 crossed above EMA26",0.6))
    if prev["ema_12"] >= prev["ema_26"] and latest["ema_12"] < latest["ema_26"]:
        signals.append(("SELL","EMA12 crossed below EMA26",0.6))
    # RSI
    if latest["rsi_14"] is not None:
        if latest["rsi_14"] < 30: signals.append(("BUY","RSI oversold",0.4))
        if latest["rsi_14"] > 75: signals.append(("SELL","RSI overbought",0.4))
    # Volume + A/D
    if latest["ad"] > df["ad"].iloc[-6] and latest["volume"] > 1.5 * latest["vol_sma_20"]:
        signals.append(("BUY","A/D rising with volume spike",0.45))
    # Elliott wave3
    piv = zigzag(df, pct=0.02)
    w3 = detect_wave3(piv)
    if w3.get("detected"):
        signals.append(("BUY",f"Elliott wave-3 detected ratio {w3['ratio']:.2f}", w3["confidence"]))
    if not signals: return None
    # aggregate
    score = {"BUY":0.0,"SELL":0.0}
    reasons = {"BUY":[],"SELL":[]}
    for s,reason,c in signals:
        score[s]+=c
        reasons[s].append(f"{reason}({c:.2f})")
    chosen = max(score.items(), key=lambda x: x[1])[0]
    total_conf = min(1.0, score[chosen])  # coarse aggregator
    reason_text = "; ".join(reasons[chosen]) if reasons[chosen] else "rules"
    return {"signal":chosen, "confidence":total_conf, "reason":reason_text, "pivots":piv}

# ----- News sources & analysis -----
NEWSAPI_URL = "https://newsapi.org/v2/everything"
GDELT_GKG = "https://api.gdeltproject.org/api/v2/doc/doc"  # fallback (we will use GDELT GKG query)
# keywords mapping to assets influences
IMPACT_KEYWORDS = {
    # gold/silver
    "gold": ["gold","xau","precious metal","safe haven"],
    "silver": ["silver","xag"],
    # oil
    "oil": ["oil","opec","barrel","brent","wti","petroleum","energy"],
    # dollar
    "dollar": ["dollar","usd","dxy","fed","federal reserve","interest rate","inflation"],
    # crypto
    "crypto": ["crypto","bitcoin","btc","ethereum","eth","coin","exchange","zk","blockchain","ban crypto","regulation"],
    # general risk-off
    "risk_off": ["war","attack","conflict","sanction","collapse","bankruptcy","default","earthquake","hurricane"]
}
# sentiment keyword scores (simple lexicon)
POSITIVE_KWS = ["boost","gain","rise","cut","easing","support","rally","surge","surprising beat","beat expectations"]
NEGATIVE_KWS = ["fall","drop","hit","attack","ban","sanction","conflict","collapse","crash","recession","hike","surge in cases","spike in deaths"]

def map_article_to_assets(title, description):
    txt = (title + " " + (description or "")).lower()
    assets = set()
    impact = []
    for asset, kws in IMPACT_KEYWORDS.items():
        for kw in kws:
            if kw in txt:
                assets.add(asset)
                impact.append(asset)
                break
    # if no direct asset keyword but words like "fed" => dollar
    if not assets:
        if "fed" in txt or "federal reserve" in txt or "rate" in txt:
            assets.add("dollar"); impact.append("dollar")
    return list(assets), list(set(impact))

def simple_sentiment(title, description):
    txt = (title + " " + (description or "")).lower()
    score = 0.0
    for kw in POSITIVE_KWS:
        if kw in txt: score += 0.6
    for kw in NEGATIVE_KWS:
        if kw in txt: score -= 0.6
    # normalize roughly
    if score > 0: return min(1.0, score)
    if score < 0: return max(-1.0, score)
    return 0.0

# ----- Fetch NewsAPI results (en + ar, plus bloomberg via domains) -----
async def fetch_newsapi(api_key, q_keywords, page_size=20, language="en", domains=None, from_dt=None):
    params = {
        "apiKey": api_key,
        "q": " OR ".join(q_keywords) if q_keywords else None,
        "language": language,
        "pageSize": page_size,
        "sortBy": "publishedAt"
    }
    if domains:
        params["domains"] = domains
    if from_dt:
        params["from"] = from_dt.isoformat()
    async with aiohttp.ClientSession() as s:
        async with s.get(NEWSAPI_URL, params={k:v for k,v in params.items() if v}) as resp:
            try:
                data = await resp.json()
            except Exception as e:
                logger.warning("NewsAPI response json error: %s", e); return []
    return data.get("articles", [])

# ----- Fetch GDELT (simple GKG query for keywords) -----
async def fetch_gdelt_gkg(keywords, timespan_minutes=30):
    # GDELT doc API can be queried for keywords; here we craft a simple search
    # Use the GDELT 2.0 Doc API: https://blog.gdeltproject.org/gdelt-2-0-our-planetary-scale-collection-of-open-source-data/
    # We'll query using 'query' param; note: GDELT limits/latency vary
    q = " OR ".join(keywords)
    params = {"query": q, "mode": "ArtList", "maxrecords": 10}
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    async with aiohttp.ClientSession() as s:
        async with s.get(url, params=params) as resp:
            try:
                data = await resp.json()
            except Exception as e:
                logger.warning("GDELT json error: %s", e); return []
    # data['articles'] may exist; if not, return empty
    return data.get("articles", []) if isinstance(data, dict) else []

# ----- News worker: analyze and generate news alerts -----
async def news_worker(app, chat_id, cfg):
    api_key = os.getenv("NEWSAPI_KEY", "")
    check_interval = cfg.get("news", {}).get("check_interval", 300)
    keywords = cfg.get("news", {}).get("keywords", ["gold","oil","fed","bitcoin","crypto","war","dxy","inflation"])
    bloomberg_domains = cfg.get("news", {}).get("bloomberg_domains", "bloomberg.com")
    last_checked = datetime.utcnow() - timedelta(minutes=30)
    while True:
        try:
            from_dt = last_checked - timedelta(seconds=5)
            # 1) NewsAPI English (including bloomberg filtered)
            if api_key:
                arts_en = await fetch_newsapi(api_key, keywords, page_size=40, language="en", domains=bloomberg_domains, from_dt=from_dt)
                arts_ar = await fetch_newsapi(api_key, keywords, page_size=40, language="ar", from_dt=from_dt)
            else:
                arts_en = []; arts_ar = []
            # 2) GDELT quick fetch
            gdelt_results = await fetch_gdelt_gkg(keywords)
            # Process NewsAPI articles
            processed_urls = set()
            for art in (arts_en + arts_ar):
                url = art.get("url"); title = art.get("title",""); desc = art.get("description","")
                lang = art.get("language","en") if art.get("language") else ("ar" if art in arts_ar else "en")
                if not url or url in processed_urls: continue
                processed_urls.add(url)
                assets,impact = map_article_to_assets(title,desc)
                sentiment = simple_sentiment(title,desc)
                # map sentiment & impact to market action
                impact_label = "neutral"
                if "risk_off" in impact or any(k in title.lower() for k in ["war","attack","conflict","sanction","earthquake","hurricane","bankrupt","default"]):
                    impact_label = "risk_off"
                elif sentiment > 0.2:
                    impact_label = "positive"
                elif sentiment < -0.2:
                    impact_label = "negative"
                # if no assets detected, fallback to mapping by keywords
                if not assets:
                    assets = impact or ["dollar"]
                save_event("newsapi", title, url, lang, impact_label, assets, sentiment)
                # Compose Telegram alert
                assets_display = ", ".join(assets)
                emot = "⚠" if impact_label in ["risk_off","negative"] else "ℹ"
                txt = (f"{emot} News Alert ({'AR' if lang=='ar' else 'EN'})\n"
                       f"Source: {art.get('source',{}).get('name','NewsAPI')}\n"
                       f"Title: {html.escape(title)}\n"
                       f"Assets: {assets_display}\n"
                       f"Impact: {impact_label} | Sentiment: {sentiment:.2f}\n"
                       f"{art.get('url')}")
                await send_msg(app, chat_id, txt)
                # adjust global signals later via shared storage (we keep it simple here)
            # Process GDELT results (if any)
            for g in gdelt_results:
                try:
                    title = g.get("title", g.get("seendate","GDELT item"))
                    url = g.get("url") or g.get("seendate")
                    assets,impact = map_article_to_assets(title, "")
                    sentiment = simple_sentiment(title, "")
                    save_event("gdelt", title, url, "en", "risk_off" if "war" in title.lower() else "neutral", assets, sentiment)
                    txt = (f"🌐 GDELT Alert\n*Title*: {html.escape(title)}\n*Assets*: {','.join(assets)}\n*Sentiment*: {sentiment:.2f}\n{url}")
                    await send_msg(app, chat_id, txt)
                except Exception:
                    continue
            last_checked = datetime.utcnow()
        except Exception as e:
            logger.exception("news worker error: %s", e)
        await asyncio.sleep(check_interval)

# ----- Main combined worker (market + technical + integrate news signals simply) -----
async def worker(app, chat_id, cfg):
    exid = cfg.get("exchange_id","binance")
    poll = cfg.get("poll_interval_seconds", 60)
    risk = cfg.get("risk_params", {"tp_factor":3, "sl_factor":1.5})
    spoof_threshold = cfg.get("spoof_threshold_usd", 100000)
    # Note: News adjustments could be pulled from DB events table for integration
    while True:
        for item in cfg.get("watchlist", []):
            pair = item["symbol"]; tf = item.get("timeframe", cfg.get("timeframe","5m"))
            try:
                df = await fetch_ohlcv(exid, pair, tf, limit=500)
                df = add_indicators(df)
                tech = evaluate_technical(df, pair)
                spoof = await detect_spoofing(exid, pair, threshold_usd=spoof_threshold)
                price = df["close"].iloc[-1]
                if tech:
                    entry = price
                    atr = df["atr_14"].iloc[-1] if not math.isnan(df["atr_14"].iloc[-1]) else price*0.005
                    if tech["signal"] == "BUY":
                        stop = price - risk["sl_factor"] * atr
                        tp = price + risk["tp_factor"] * atr
                    else:
                        stop = price + risk["sl_factor"] * atr
                        tp = price - risk["tp_factor"] * atr
                    conf = tech["confidence"]
                    reason = tech["reason"]
                    # News integration heuristic: check recent events in DB that affect this pair
                    # simple approach: scan last N events for this asset keywords
                    conn = sqlite3.connect(DB); cur = conn.cursor()
                    cur.execute("SELECT impact,assets,sentiment FROM events ORDER BY id DESC LIMIT 30")
                    recent = cur.fetchall(); conn.close()
                    news_boost = 0.0
                    if recent:
                        for ev in recent:
                            ev_assets = ev[1].split(",") if ev[1] else []
                            if any(a in ["crypto"] for a in ev_assets) and pair.lower().startswith(("btc","eth","bnb","ada")):
                                # if negative sentiment and risk_off -> reduce conf
                                if ev[0] in ("risk_off","negative") and ev[2] < -0.1:
                                    conf *= 0.7; reason += f"; news({ev[0]})"
                                elif ev[0] in ("positive") and ev[2] > 0.1:
                                    conf = min(1.0, conf * 1.15); reason += f"; news({ev[0]})"
                            if "gold" in ev_assets and pair.upper().startswith(("XAU","GOLD","XAG")):
                                if ev[0]=="risk_off": conf = min(1.0, conf*1.2); reason += "; news risk_off->gold"
                    # spoof handling
                    if spoof:
                        if (tech["signal"]=="BUY" and spoof["side"]=="buy") or (tech["signal"]=="SELL" and spoof["side"]=="sell"):
                            conf *= 0.6
                            reason += f"; spoofing {spoof['side']} {spoof['value']:.0f}USD"
                    # save and send
                    save_signal(pair, tf, tech["signal"], entry, stop, tp, conf, reason)
                    msg = (f"Signal: {tech['signal']}\n*Pair*: {pair} ({tf})\n"
                           f"Entry: {entry:.8f}\n*Stop*: {stop:.8f}\n*TP*: {tp:.8f}\n"
                           f"Confidence: {conf:.2f}\n*Reason*: {reason}\n*Time (UTC)*: {df.index[-1]}")
                    await send_msg(app, chat_id, msg)
                await asyncio.sleep(1)
            except Exception as e:
                logger.exception("worker pair error: %s %s", pair, e)
                await asyncio.sleep(3)
        await asyncio.sleep(poll)

# ----- Telegram commands -----
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Botsignal: جاهز. استخدم /status و /last و /events")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    cfg = load_config()
    await update.message.reply_text(f"Monitoring {len(cfg.get('watchlist',[]))} pairs. Poll {cfg.get('poll_interval_seconds')}s")

async def last(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = sqlite3.connect(DB); cur = conn.cursor()
    cur.execute("SELECT ts,pair,signal,entry,tp,stop,confidence FROM signals ORDER BY id DESC LIMIT 10")
    rows = cur.fetchall(); conn.close()
    if not rows:
        await update.message.reply_text("No signals yet.")
        return
    msg = "Last signals:\n"
    for r in rows:
        msg += f"{r[0]} {r[1]} {r[2]} @ {r[3]:.6f} TP:{r[4]:.6f} SL:{r[5]:.6f} ({r[6]:.2f})\n"
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def events_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    conn = sqlite3.connect(DB); cur = conn.cursor()
    cur.execute("SELECT ts,source,title,impact,assets,sentiment FROM events ORDER BY id DESC LIMIT 10")
    rows = cur.fetchall(); conn.close()
    if not rows:
        await update.message.reply_text("No events yet.")
        return
    msg = "Recent events:\n"
    for r in rows:
        msg += f"{r[0]} | {r[1]} | {r[3]} | assets:{r[4]} | s:{r[5]:.2f}\n{r[2]}\n\n"
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

# ----- Main -----
async def main():
    init_db()
    cfg = load_config()
    TOKEN = os.getenv("TG_BOT_TOKEN"); CHAT = int(os.getenv("TG_CHAT_ID","0"))
    if not TOKEN or CHAT==0:
        raise RuntimeError("Set TG_BOT_TOKEN and TG_CHAT_ID env vars")
    app = ApplicationBuilder().token(TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("last", last))
    app.add_handler(CommandHandler("events", events_cmd))
    # start workers
    asyncio.create_task(news_worker(app, CHAT, cfg))
    asyncio.create_task(worker(app, CHAT, cfg))
    await app.run_polling()

if __name__ == "__main__":
    import asyncio
    loop = asyncio.get_event_loop()
    loop.create_task(main())
    loop.run_forever()    
# --- Test message to verify Telegram connection ---
import asyncio
from telegram import Bot

async def test_message():
    bot = Bot(token=os.getenv("TG_BOT_TOKEN"))
    await bot.send_message(chat_id=os.getenv("TG_CHAT_ID"), text="🚀 Bot is live and connected successfully!")

asyncio.run(test_message())



