import { useEffect, useRef } from "react";

export default function ChatBox({
  messages = [], 
  onClear
}) {
  const chatEndRef = useRef(null);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [messages]);

  return (
    <div className="chat-window">

      {messages.length > 0 && (
        <div style={{ textAlign: "center", marginBottom: 12 }}>
          <button className="clear-btn-in-chat" onClick={onClear}>
            기록삭제
          </button>
        </div>
      )}


      {messages.map((msg, i) => (
        <div key={i} className="chat-message-block">
          <div className="chat-bubble user">🧑‍💻 {msg.query}</div>
          <div className="chat-bubble ai">
            <div className="trend-summary">
              <h4>✨ {msg.results.purpose} 트렌드 요약</h4>
              <p>{msg.results.trend_digest}</p>
            </div>
            <hr className="divider" />

            {msg.results.trend_articles?.map((article, j) => (
              <div key={j} className="news-card">
                <h4>{article.title}</h4>
                <p>{article.summary}</p>
                <a href={article.url} target="_blank" rel="noreferrer">
                  기사 원문 보기
                </a>
              </div>
            ))}
          </div>
        </div>
      ))}

      <div ref={chatEndRef} />
    </div>
  );
}
