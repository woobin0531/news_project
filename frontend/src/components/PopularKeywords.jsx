import { useEffect, useState } from "react";

const API_URL = "http://localhost:8000"; 

export default function PopularKeywords({ onSelect }) {
  const [keywords, setKeywords] = useState([]);
  const [error, setError] = useState("");

  useEffect(() => {
    (async () => {
      try {
        const res = await fetch(`${API_URL}/popular_keywords`, {
          headers: { Accept: "application/json" },
        });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json(); 
        if (!Array.isArray(data)) throw new Error("Invalid JSON");
        setKeywords(data.map((d) => d.keyword));
      } catch (e) {
        console.error("인기 키워드 불러오기 실패:", e);
        setError("인기 키워드를 불러오지 못했습니다.");
      }
    })();
  }, []);

  if (error) return <div style={{ color: "#d33" }}>{error}</div>;
  if (!keywords.length) return <div>인기 키워드 로딩 중…</div>;

  return (
    <div className="popular-keywords">
      {keywords.map((k) => (
        <button key={k} onClick={() => onSelect?.(k)} className="keyword-btn">
          {k}
        </button>
      ))}
    </div>
  );
}
