import { Moon, Sun } from "lucide-react";
import { Button } from "@/components/ui/button";
import { useTheme } from "@/lib/theme";

/** Fixed to the corner rather than sitting in a header, because this app has no header --
 * every page is a standalone column. Rendered once in App so it's on all of them. */
export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const next = theme === "dark" ? "light" : "dark";

  return (
    <Button
      variant="ghost"
      size="sm"
      className="fixed right-4 top-4 z-50 h-9 w-9 p-0 text-muted-foreground"
      onClick={() => setTheme(next)}
      // The label says what the click does, not what the icon is -- a screen reader user
      // gets no information from "sun".
      aria-label={`Switch to ${next} mode`}
      title={`Switch to ${next} mode`}
    >
      {theme === "dark" ? (
        <Sun className="h-4 w-4" aria-hidden />
      ) : (
        <Moon className="h-4 w-4" aria-hidden />
      )}
    </Button>
  );
}
