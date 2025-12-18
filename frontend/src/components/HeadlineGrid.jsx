import { useEffect, useState } from "react";

const RAW = (process.env.REACT_APP_API_URL || "http://localhost:8000").trim().replace(/\/+$/, "");
const API_URL = RAW.endsWith("/api") ? RAW : `${RAW}/api`;


const fetchJSON = async (url, options = {}, ms = 10000) => {
  const ctrl = new AbortController();
  const t = setTimeout(() => ctrl.abort(), ms);
  try {
    const res = await fetch(url, {
      ...options,
      signal: ctrl.signal,
      headers: { Accept: "application/json", ...(options.headers || {}) },
      mode: "cors",
      credentials: "omit",
      cache: "no-store",
      redirect: "follow",
    });
    const ct = (res.headers.get("content-type") || "").toLowerCase();
    const text = await res.text();

    if (res.ok && ct.includes("application/json")) return JSON.parse(text);

    if (ct.includes("application/json")) {
      let payload; try { payload = JSON.parse(text); } catch {}
      const msg = payload?.detail || payload?.message || text.slice(0,160);
      throw new Error(`HTTP ${res.status}: ${msg}`);
    }

    if (ct.includes("text/html")) {
      const [base, qs] = url.split("?", 2);
      const flippedBase = base.endsWith("/") ? base.slice(0, -1) : `${base}/`;
      const alt = qs ? `${flippedBase}?${qs}` : flippedBase;

      const r2 = await fetch(alt, {
        ...options,
        signal: ctrl.signal,
        headers: { Accept: "application/json", ...(options.headers || {}) },
        mode: "cors",
        credentials: "omit",
        cache: "no-store",
        redirect: "follow",
      });
      const ct2 = (r2.headers.get("content-type") || "").toLowerCase();
      const tx2 = await r2.text();

      if (r2.ok && ct2.includes("application/json")) return JSON.parse(tx2);
      if (ct2.includes("application/json")) {
        let payload; try { payload = JSON.parse(tx2); } catch {}
        const msg = payload?.detail || payload?.message || tx2.slice(0,160);
        throw new Error(`HTTP ${r2.status}: ${msg}`);
      }
      throw new Error(`JSON 아님: ${ct2 || "unknown"}`);
    }

    throw new Error(`HTTP ${res.status} ${text.slice(0,160)}…`);
  } finally {
    clearTimeout(t);
  }
};

export default function HeadlineGrid({ keywords = [], onSearch }) {
  const [cards, setCards] = useState(
    keywords.map((k) => ({ kw: k, state: "loading", data: null, error: null }))
  );

  useEffect(() => {
    let gone = false;
    setCards(keywords.map((k) => ({ kw: k, state: "loading", data: null, error: null })));

    (async () => {
      const jobs = keywords.map(async (k) => {
        try {
          const url = `${API_URL}/headline_quick?kw=${encodeURIComponent(k)}`;
          const data = await fetchJSON(url, {}, 12000);
          return { kw: k, state: "ok", data, error: null };
        } catch (e) {
          console.warn(`[headline_quick 실패] ${k}:`, e.message);
          return { kw: k, state: "fail", data: null, error: e.message || "fail" };
        }
      });

      const results = await Promise.all(jobs);
      if (!gone) setCards(results);
    })();

    return () => { gone = true; };
  }, [keywords]);

  return (
    <div className="cards-grid" style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 16 }}>
      {cards.map(({ kw, state, data }, i) => (
        <article key={kw + i} className="card" style={{ border: "1px solid #eee", borderRadius: 12, padding: 16, background: "#fff" }}>
          <div style={{ fontSize: 13, color: "#999", marginBottom: 6 }}>🔥 {kw}</div>

          {state === "loading" && (
            <>
              <strong>(로딩 중)</strong>
              <p style={{ marginTop: 6, color: "#888" }}>요약 준비 중…</p>
            </>
          )}

          {state === "fail" && (
            <>
              <strong>(로딩 실패)</strong>
              <p style={{ marginTop: 6, color: "#888" }}>요약 준비 중…</p>
            </>
          )}

          {state === "ok" && data && (
            <>
              <h4 style={{ margin: "8px 0" }}>
                {data.url ? (
                  <a href={data.url} target="_blank" rel="noreferrer">{data.title || "(제목 없음)"}</a>
                ) : (data.title || "(제목 없음)")}
              </h4>
              <p style={{ margin: 0, color: "#444" }}>{data.summary || ""}</p>
            </>
          )}

          <div style={{ marginTop: 12 }}>
            <button
              type="button"
              onClick={() => (onSearch ? onSearch(kw, { fastPreview: false }) : null)}
              className="btn"
              title={`${kw} 상세 요약 보기`}
              style={{ padding: "6px 10px", borderRadius: 8, border: "1px solid #ddd", background: "#fff", cursor: "pointer" }}
            >
              더 보기 →
            </button>
          </div>
        </article>
      ))}
    </div>
  );
}
