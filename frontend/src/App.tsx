import { Suspense, lazy } from "react";
import { Navigate, Route, Routes } from "react-router-dom";
import { ThemeToggle } from "@/components/ui/theme-toggle";
import { UploadPage } from "@/pages/UploadPage";
import { ReviewPage } from "@/pages/ReviewPage";
import { CleanPage } from "@/pages/CleanPage";
import { ExportPage } from "@/pages/ExportPage";
import { ApplyPage } from "@/pages/ApplyPage";
import { ProjectBar } from "@/components/project/ProjectBar";
import { RouteFocus } from "@/components/ui/route-focus";

// Split out: the model page is the only thing that pulls in the charting library, and it
// roughly doubles the bundle. Someone who only ever cleans a CSV should not download it.
const ModelPage = lazy(() =>
  import("@/pages/ModelPage").then((module) => ({ default: module.ModelPage }))
);

export default function App() {
  return (
    <>
      {/* Landmarks, so a screen reader can jump straight to the page's content (#101). */}
      <header>
        <ThemeToggle />
        {/* On every page of a session: keep it as a project, rename it, delete it (#120). */}
        <Routes>
          <Route path="/:step/:sessionId" element={<ProjectBar />} />
          <Route path="*" element={null} />
        </Routes>
      </header>
      <RouteFocus />
      <main id="main">
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
        <Route path="/apply/:sessionId" element={<ApplyPage />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
      </main>
    </>
  );
}
