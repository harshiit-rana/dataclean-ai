/** Scores as people read them (#132).
 *
 * Models with more than two classes are chosen by log loss. The backend compares it as its negative,
 * sklearn's "neg_log_loss", so that higher is better everywhere; on screen it is the positive number
 * people know, where lower is better. A gain is the change in that shown number. */
export const LOG_LOSS = "neg_log_loss";

export function shownScore(value: number, metric?: string | null): number {
  return metric === LOG_LOSS ? -value : value;
}

/** A change, signed, in the shown number: a log loss gain shows as the log loss going down. */
export function shownGain(gain: number, metric?: string | null): string {
  const value = shownScore(gain, metric);
  return `${value >= 0 ? "+" : ""}${value.toFixed(3)}`;
}
