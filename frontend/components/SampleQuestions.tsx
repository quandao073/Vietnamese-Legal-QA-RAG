export const SAMPLE_QUESTIONS = [
  "Đối tượng nào không chịu thuế giá trị gia tăng?",
  "Thu nhập nào được miễn thuế thu nhập cá nhân?",
  "Mức thuế suất thuế thu nhập doanh nghiệp phổ thông là bao nhiêu?",
  "Ai là người nộp thuế sử dụng đất phi nông nghiệp?",
];

export default function SampleQuestions({
  onPick,
  disabled,
}: {
  onPick: (q: string) => void;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-col items-center gap-3 py-10 text-center">
      <h2 className="text-lg font-semibold">Bạn muốn hỏi gì về pháp luật thuế?</h2>
      <p className="max-w-md text-sm text-slate-500 dark:text-slate-400">
        Trợ lý trả lời dựa trên kho văn bản pháp luật và luôn kèm trích dẫn tới điều khoản gốc.
      </p>
      <div className="mt-2 flex flex-wrap justify-center gap-2">
        {SAMPLE_QUESTIONS.map((q) => (
          <button
            key={q}
            type="button"
            disabled={disabled}
            onClick={() => onPick(q)}
            className="rounded-full border border-slate-300 bg-white px-3 py-1.5 text-sm text-slate-700 hover:border-blue-400 hover:text-blue-700 disabled:opacity-50 dark:border-slate-700 dark:bg-slate-900 dark:text-slate-200 dark:hover:border-blue-500 dark:hover:text-blue-300"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
