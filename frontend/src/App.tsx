import { Navigate, Route, Routes } from "react-router-dom";
import { UploadPage } from "@/pages/UploadPage";
import { ReviewPage } from "@/pages/ReviewPage";
import { CleanPage } from "@/pages/CleanPage";
import { ExportPage } from "@/pages/ExportPage";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/review/:sessionId" element={<ReviewPage />} />
      <Route path="/clean/:sessionId" element={<CleanPage />} />
      <Route path="/export/:sessionId" element={<ExportPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
