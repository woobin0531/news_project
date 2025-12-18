export default function ListeningLoader({ image, text }) {
  return (
    <div className="loader listening-loader">
      <img src={image} alt="listening hamster" className="loader-img" />
      <p>{text}</p>
    </div>
  );
}
