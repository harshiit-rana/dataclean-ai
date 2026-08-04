import { Suspense, lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { UploadPage } from "@/pages/UploadPage";
import { ReviewPage } from "@/pages/ReviewPage";
import { CleanPage } from "@/pages/CleanPage";
import { ExportPage } from "@/pages/ExportPage";

// Split out: the model page is the only thing that pulls in the charting library, and it
// roughly doubles the bundle. Someone who only ever cleans a CSV should not download it.
const ModelPage = lazy(() =>
  import("@/pages/ModelPage").then((module) => ({ default: module.ModelPage }))
);

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<UploadPage />} />
      <Route path="/review/:sessionId" element={<ReviewPage />} />
      <Route path="/clean/:sessionId" element={<CleanPage />} />
      <Route
        path="/model/:sessionId"
        element={
          <Suspense
            fallback={
              <div className="mx-auto max-w-3xl p-8 text-sm text-muted-foreground">
                Loading the model builder...
              </div>
            }
          >
            <ModelPage />
          </Suspense>
        }
      />
      <Route path="/export/:sessionId" element={<ExportPage />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
}
