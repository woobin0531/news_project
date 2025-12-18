export default function Loader({ videoSrc, text }) {
  return (
    <div className="loader">
      <video
        src={videoSrc}
        className="loader-img"
        autoPlay
        loop
        muted
        playsInline
      />
      <p>{text || "검색 중입니다..."}</p>
    </div>
  );
}