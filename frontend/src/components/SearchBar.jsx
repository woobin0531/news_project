import { useState } from "react";

export default function SearchBar({ onSearch, onRecordingStart, onRecordingEnd }) {
  const [text, setText] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(text);
    setText("");
  };

  const handleSTT = () => {
    const recognition = new window.webkitSpeechRecognition();
    recognition.lang = "ko-KR";

    if (onRecordingStart) onRecordingStart();
    recognition.start();

    recognition.onresult = (e) => {
      const transcript = e.results[0][0].transcript;
      setText("");
      onSearch(transcript);
    };

    recognition.onend = () => {
      if (onRecordingEnd) onRecordingEnd();
    };

    recognition.onerror = () => {
      alert("음성 인식 실패");
      if (onRecordingEnd) onRecordingEnd();
    };
  };

  return (
    <form className="search-bar" onSubmit={handleSubmit}>
      <input
        type="text"
        placeholder="무엇이든 물어보세요"
        value={text}
        onChange={(e) => setText(e.target.value)}
      />
      <button type="submit">검색</button>
      <button type="button" onClick={handleSTT}>🎤</button>
    </form>
  );
}
