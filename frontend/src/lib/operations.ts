/** Plain-English labels for the whitelisted cleaning operations in
 * backend/app/services/operations.py. Shared by the live progress view and the persisted
 * action log so the same fix never gets described two different ways. */
export const OPERATION_LABELS: Record<string, string> = {
  fill_median: "Filled missing values with the median",
  fill_mean: "Filled missing values with the mean",
  fill_mode: "Filled missing values with the most common value",
  fill_constant: "Filled missing values with a fixed value",
  drop_rows_with_nulls: "Dropped rows with missing values",
  drop_duplicate_rows: "Removed duplicate rows",
  cast_to_numeric: "Converted to numbers",
  cast_to_datetime: "Converted to dates",
  standardize_case: "Made capitalization consistent",
  trim_whitespace: "Removed extra spaces",
  standardize_dates: "Standardized date formats",
  cap_outliers: "Clamped extreme values",
  drop_outlier_rows: "Dropped rows with extreme values",
  drop_column: "Removed the column",
  // The four chat instructions map onto. Missing until the apply page listed a session's
  // steps -- any view of a chat-cleaned log showed these as raw operation names.
  rename_column: "Renamed the column",
  round_numeric: "Rounded the numbers",
  replace_values: "Replaced one value with another",
  filter_rows: "Kept or dropped rows matching a condition",
  none: "No executable fix was attached",
};
