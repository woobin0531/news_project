export default function MicButton({ onSend }) {
  const handleClick = () => {
    const recognition = new window.webkitSpeechRecognition();
    recognition.lang = "ko-KR";
    recognition.start();

    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      onSend(transcript);
    };

    recognition.onerror = () => alert("음성 인식 실패");
  };

  return (
    <button className="mic-button" onClick={handleClick}>
      🎤 마이크로 질문하기
    </button>
  );
}
