import * as React from "react";
import { cn } from "@/lib/utils";

type Variant = "default" | "high" | "medium" | "low";

const variantClasses: Record<Variant, string> = {
  default: "bg-muted text-muted-foreground",
  high: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
  medium: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  low: "bg-blue-100 text-blue-800 dark:bg-blue-950 dark:text-blue-300",
};

export function Badge({
  className,
  variant = "default",
  ...props
}: React.HTMLAttributes<HTMLSpanElement> & { variant?: Variant }) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        variantClasses[variant],
        className
      )}
      {...props}
    />
  );
}
