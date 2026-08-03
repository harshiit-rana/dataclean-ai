import { useState } from "react";
import type { GoalContext, UploadResponse } from "@/api/types";

/** Holds the session created by Page 1 (upload + goal context) so Page 2 can read it via
 * route state / this hook's return value without a global store -- the scaffold only needs
 * one session alive at a time (US-01..US-04). */
export function useSession() {
  const [upload, setUpload] = useState<UploadResponse | null>(null);
  const [goalContext, setGoalContext] = useState<GoalContext | null>(null);

  return { upload, setUpload, goalContext, setGoalContext };
}
