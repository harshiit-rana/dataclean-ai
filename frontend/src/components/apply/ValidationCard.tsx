import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import type { FileValidation } from "@/api/types";

/** The new file checked against what the model was trained on, column by column.
 *
 * Which rows to look at is the point: a count alone sends the user hunting through the file.
 * Every finding here is a warning -- the rows were still scored -- because a missing column,
 * the one blocking problem, stops the run before this is shown. */

const KIND_LABEL: Record<string, string> = {
  type: "Wrong type",
  unseen_category: "Values not seen in training",
  out_of_range: "Outside the training range",
  blank_new: "Blank where training never was",
};

export function ValidationCard({ validation }: { validation: FileValidation }) {
  const clean = validation.findings.length === 0 && validation.unexpected_columns.length === 0;
  return (
    <Card>
      <CardHeader>
        <CardTitle className="text-base">
          {clean
            ? "The file matches what the model was trained on"
            : `${validation.rows_with_findings.toLocaleString()} of ${validation.rows_checked.toLocaleString()} rows have something to check`}
        </CardTitle>
        <p className="text-sm text-muted-foreground">
          {clean
            ? "Every column the model needs is here, with the kinds of values it learned from."
            : "They were still scored. The predictions file has a dataclean_validation column saying what each row's problem is."}
        </p>
      </CardHeader>
      {!clean && (
        <CardContent className="flex flex-col gap-3 text-sm">
          {validation.unexpected_columns.length > 0 && (
            <p className="text-muted-foreground">
              Not used by the model, so ignored: {validation.unexpected_columns.join(", ")}.
            </p>
          )}
          {validation.findings.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="text-left text-muted-foreground">
                    <th className="py-1 pr-3 font-medium">Column</th>
                    <th className="py-1 pr-3 font-medium">Problem</th>
                    <th className="py-1 pr-3 font-medium">Rows</th>
                    <th className="py-1 font-medium">For example</th>
                  </tr>
                </thead>
                <tbody>
                  {validation.findings.map((f) => (
                    <tr key={`${f.column}-${f.finding}`} className="border-t border-border/60 align-top">
                      <td className="py-1 pr-3 font-medium">{f.column}</td>
                      <td className="py-1 pr-3">
                        {KIND_LABEL[f.finding] ?? f.finding}
                        <span className="block text-muted-foreground">{f.message}</span>
                      </td>
                      <td className="py-1 pr-3 tabular-nums">{f.count.toLocaleString()}</td>
                      <td className="py-1">
                        rows {f.rows.join(", ")}
                        {f.count > f.rows.length && " …"}
                        {f.values && f.values.length > 0 && (
                          <span className="block text-muted-foreground">{f.values.map((v) => `“${v}”`).join(", ")}</span>
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <p className="mt-1 text-xs text-muted-foreground">Row numbers count the file&apos;s rows, not its header.</p>
            </div>
          )}
        </CardContent>
      )}
    </Card>
  );
}
