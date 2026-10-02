"""
Универсальный сигнальный бот: крипта (ccxt/Binance), акции и форекс (yfinance).
Символы: BTC/USDT (крипта), AAPL (акции), EURUSD=X (форекс), GC=F (золото).
Таймфреймы: 15m, 1h, 4h, 1d.
Стратегия: тренд по EMA50/EMA200 + вход на откате по RSI, стоп/тейк по ATR.
Это не гарантия прибыли. Всегда смотрите /backtest и рискуйте не более 1% депозита.
"""
import asyncio
import logging
import os

import ccxt
import numpy as np
import pandas as pd
import yfinance as yf
from aiogram import Bot, Dispatcher
from aiogram.filters import Command, CommandObject
from aiogram.types import Message

TOKEN = os.getenv("BOT_TOKEN")
SL_ATR, TP_ATR = 1.5, 3.0      # стоп 1.5 ATR, тейк 3 ATR (RR = 1:2)
FEE_R = 0.05                   # комиссии/проскальзывание в долях риска на сделку
CHECK_EVERY = 60               # секунд между проверками /watch
PERIOD = {"15m": "60d", "1h": "730d", "4h": "730d", "1d": "5y"}

exchange = ccxt.binance({"enableRateLimit": True})
watch: dict[tuple, int] = {}   # (chat_id, symbol, tf) -> метка последней отправленной свечи


# ---------- данные ----------
def fetch(symbol: str, tf: str) -> pd.DataFrame:
    if "/" in symbol:
        raw = exchange.fetch_ohlcv(symbol, tf, limit=1000)
        df = pd.DataFrame(raw, columns=["ts", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df.pop("ts"), unit="ms")
    else:
        df = yf.download(symbol, period=PERIOD[tf], interval="1h" if tf == "4h" else tf,
                         progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)[["open", "high", "low", "close"]].dropna()
        if tf == "4h":
            df = df.resample("4h").agg({"open": "first", "high": "max",
                                        "low": "min", "close": "last"}).dropna()
    if len(df) < 250:
        raise ValueError("мало данных или неверный тикер")
    return df.iloc[:-1]  # последняя свеча ещё не закрыта


def add_ind(df: pd.DataFrame) -> pd.DataFrame:
    c = df.close
    df["ema50"] = c.ewm(span=50, adjust=False).mean()
    df["ema200"] = c.ewm(span=200, adjust=False).mean()
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
    tr = pd.concat([df.high - df.low, (df.high - c.shift()).abs(),
                    (df.low - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    return df


def signals(df: pd.DataFrame) -> pd.Series:
    long_ = (df.close > df.ema200) & (df.ema50 > df.ema200) & (df.rsi.shift() < 45) & (df.rsi >= 45)
    short = (df.close < df.ema200) & (df.ema50 < df.ema200) & (df.rsi.shift() > 55) & (df.rsi <= 55)
    return long_.astype(int) - short.astype(int)


# ---------- бэктест ----------
def backtest(df: pd.DataFrame) -> list[float]:
    df = add_ind(df).dropna()
    sig = signals(df)
    o, h, l, a = df.open.values, df.high.values, df.low.values, df.atr.values
    rs, i, n = [], 0, len(df)
    while i < n - 1:
        s = sig.iloc[i]
        if s == 0:
            i += 1
            continue
        entry = o[i + 1]
        sl = entry - s * SL_ATR * a[i]
        tp = entry + s * TP_ATR * a[i]
        r, j = None, i + 1
        for j in range(i + 1, n):
            hit_sl = l[j] <= sl if s == 1 else h[j] >= sl
            hit_tp = h[j] >= tp if s == 1 else l[j] <= tp
            if hit_sl:  # если в одной свече оба уровня, считаем худшее
                r = -1.0
                break
            if hit_tp:
                r = TP_ATR / SL_ATR
                break
        if r is None:
            break
        rs.append(r - FEE_R)
        i = j + 1
    return rs


def stats_text(rs: list[float]) -> str:
    if not rs:
        return "Сделок не найдено."
    eq = np.cumsum(rs)
    dd = (np.maximum.accumulate(eq) - eq).max()
    wr = sum(r > 0 for r in rs) / len(rs) * 100
    return (f"Сделок: {len(rs)}\nВинрейт: {wr:.1f}%\n"
            f"Средний результат: {np.mean(rs):+.2f}R\nИтого: {eq[-1]:+.1f}R\n"
            f"Макс. просадка: {dd:.1f}R\n(R = размер риска на сделку)")


# ---------- сигнал ----------
def check(symbol: str, tf: str):
    df = add_ind(fetch(symbol, tf)).dropna()
    s = signals(df).iloc[-1]
    ts = int(df.index[-1].timestamp())
    if s == 0:
        return None, ts
    price, atr = df.close.iloc[-1], df.atr.iloc[-1]
    sl, tp = price - s * SL_ATR * atr, price + s * TP_ATR * atr
    side = "🟢 LONG" if s == 1 else "🔴 SHORT"
    return (f"{side} {symbol} [{tf}]\nВход: {price:.5g}\nСтоп: {sl:.5g}\nТейк: {tp:.5g}\n"
            f"RR 1:{TP_ATR / SL_ATR:.0f}. Риск не более 1% депозита."), ts


def parse(cmd: CommandObject):
    parts = (cmd.args or "").split()
    if not parts:
        raise ValueError("Пример: /signal BTC/USDT 1h")
    tf = parts[1] if len(parts) > 1 else "1h"
    if tf not in PERIOD:
        raise ValueError("Таймфрейм: 15m, 1h, 4h, 1d")
    return parts[0].upper(), tf


# ---------- Telegram ----------
dp = Dispatcher()


@dp.message(Command("start", "help"))
async def start(m: Message):
    await m.answer(
        "Команды:\n/signal BTC/USDT 1h — проверить сигнал сейчас\n"
        "/backtest AAPL 1d — бэктест стратегии\n/watch EURUSD=X 1h — присылать сигналы\n"
        "/unwatch EURUSD=X 1h\n/list\n\n"
        "Крипта: BTC/USDT, акции: AAPL, форекс: EURUSD=X, золото: GC=F.\n"
        "Сигналы не гарантируют прибыль.")


@dp.message(Command("signal"))
async def cmd_signal(m: Message, command: CommandObject):
    try:
        sym, tf = parse(command)
        text, _ = await asyncio.to_thread(check, sym, tf)
        await m.answer(text or f"{sym} [{tf}]: сигнала нет.")
    except Exception as e:
        await m.answer(f"Ошибка: {e}")


@dp.message(Command("backtest"))
async def cmd_bt(m: Message, command: CommandObject):
    try:
        sym, tf = parse(command)
        df = await asyncio.to_thread(fetch, sym, tf)
        rs = await asyncio.to_thread(backtest, df)
        await m.answer(f"Бэктест {sym} [{tf}]\n{stats_text(rs)}")
    except Exception as e:
        await m.answer(f"Ошибка: {e}")


@dp.message(Command("watch"))
async def cmd_watch(m: Message, command: CommandObject):
    try:
        sym, tf = parse(command)
        watch[(m.chat.id, sym, tf)] = 0
        await m.answer(f"Слежу за {sym} [{tf}].")
    except Exception as e:
        await m.answer(f"Ошибка: {e}")


@dp.message(Command("unwatch"))
async def cmd_unwatch(m: Message, command: CommandObject):
    try:
        sym, tf = parse(command)
        watch.pop((m.chat.id, sym, tf), None)
        await m.answer("Остановлено.")
    except Exception as e:
        await m.answer(f"Ошибка: {e}")


@dp.message(Command("list"))
async def cmd_list(m: Message):
    items = [f"{s} [{t}]" for (c, s, t) in watch if c == m.chat.id]
    await m.answer("\n".join(items) or "Список пуст.")


async def watcher(bot: Bot):
    while True:
        for key in list(watch):
            chat, sym, tf = key
            try:
                text, ts = await asyncio.to_thread(check, sym, tf)
                if text and watch.get(key) != ts:
                    watch[key] = ts
                    await bot.send_message(chat, text)
            except Exception as e:
                logging.warning("%s: %s", key, e)
        await asyncio.sleep(CHECK_EVERY)


async def main():
    bot = Bot(TOKEN)
    asyncio.create_task(watcher(bot))
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())