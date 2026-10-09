export const fmt = {
  int: (n: number) => Math.round(n).toLocaleString("en-US"),
  usd: (n: number) => "$" + Math.round(n).toLocaleString("en-US"),
  usd2: (n: number) =>
    "$" + n.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }),
  compactUsd: (n: number) => {
    const a = Math.abs(n);
    if (a >= 1e6) return "$" + (n / 1e6).toFixed(2) + "M";
    if (a >= 1e4) return "$" + (n / 1e3).toFixed(1) + "K";
    return fmt.usd(n);
  },
  pct: (x: number, digits = 1) => (x * 100).toFixed(digits) + "%",
  prob: (p: number) => {
    if (p >= 0.995) return ">99%";
    if (p >= 0.1) return (p * 100).toFixed(0) + "%";
    if (p >= 0.01) return (p * 100).toFixed(1) + "%";
    if (p >= 0.0001) return (p * 100).toFixed(2) + "%";
    return "<0.01%";
  },
  dateTime: (ts: number) => new Date(ts * 1000).toISOString().slice(0, 19).replace("T", " "),
  date: (ts: number) => new Date(ts * 1000).toISOString().slice(0, 10),
  time: (ts: number) => new Date(ts * 1000).toISOString().slice(11, 19),
  hour: (h: number) => String(h).padStart(2, "0") + ":00",
  duration: (s: number | null) => {
    if (s == null) return "none";
    if (s < 90) return Math.round(s) + " s";
    if (s < 5400) return Math.round(s / 60) + " min";
    if (s < 172800) return (s / 3600).toFixed(1) + " h";
    return Math.round(s / 86400) + " days";
  },
  card: (id: number) => "•••• " + String(id).slice(-4),
  ms: (x: number) => (x < 10 ? x.toFixed(2) : x.toFixed(0)) + " ms",
};
