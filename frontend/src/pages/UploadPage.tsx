import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { FileDropzone } from "@/components/upload/FileDropzone";
import { GoalForm } from "@/components/upload/GoalForm";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { ApiError, submitGoalContext, uploadCsv } from "@/api/client";
import type { GoalContext } from "@/api/types";

/** Page 1, PRD US-01/US-02: upload a CSV, describe the goal, then move to Page 2 which
 * starts profiling. */
export function UploadPage() {
  const navigate = useNavigate();
  const [file, setFile] = useState<File | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (goal: GoalContext) => {
    if (!file) {
      setError("Choose a CSV file first.");
      return;
    }
    setSubmitting(true);
    setError(null);
    try {
      const upload = await uploadCsv(file);
      await submitGoalContext(upload.session_id, goal);
      navigate(`/review/${upload.session_id}`);
    } catch (err) {
      setError(err instanceof ApiError ? err.detail : "Something went wrong. Please try again.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="mx-auto flex max-w-2xl flex-col gap-6 p-8">
      <div>
        <h1 className="text-2xl font-semibold">DataClean AI</h1>
        <p className="text-muted-foreground">
          Upload a CSV, tell us what you're trying to do with it, and we'll flag what needs
          cleaning -- you approve every fix before it runs.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>1. Upload your CSV</CardTitle>
        </CardHeader>
        <CardContent>
          <FileDropzone onFileSelected={setFile} selectedFile={file} />
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>2. Tell us your goal</CardTitle>
        </CardHeader>
        <CardContent>
          <GoalForm onSubmit={handleSubmit} submitting={submitting} />
        </CardContent>
      </Card>

      {error && <p className="text-sm text-destructive">{error}</p>}
    </div>
  );
}
