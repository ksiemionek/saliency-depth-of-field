export default function ResultViewer({ result }) {
    return (
      <div className="result-viewer">
        <img src={result.output_url} alt="Wynik przetwarzania" />
      </div>
    )
  }