import { useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { cn } from "@/lib/utils";

interface FileDropzoneProps {
  onFileSelected: (file: File) => void;
  selectedFile: File | null;
}

export function FileDropzone({ onFileSelected, selectedFile }: FileDropzoneProps) {
  const onDrop = useCallback(
    (accepted: File[]) => {
      if (accepted[0]) onFileSelected(accepted[0]);
    },
    [onFileSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { "text/csv": [".csv"] },
    multiple: false,
  });

  return (
    <>
      <div
        // A button to assistive technology: the dropzone's own role is "presentation", which left a focusable element
        // with nothing to announce (#101). Enter or Space opens the file picker.
        {...getRootProps({
          role: "button",
          "aria-label": selectedFile
            ? `${selectedFile.name} chosen. Choose a different CSV file, or drop one here`
            : "Choose a CSV file, or drop one here",
        })}
        className={cn(
          "flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border p-10 text-center transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2 focus-visible:ring-offset-background",
          isDragActive && "border-primary bg-muted"
        )}
      >
        {selectedFile ? (
          <p className="text-sm font-medium">{selectedFile.name}</p>
        ) : (
          <>
            <p className="text-sm font-medium">Drop a CSV file here, or click to browse</p>
            <p className="mt-1 text-xs text-muted-foreground">CSV only, up to 50MB</p>
          </>
        )}
      </div>
      {/* Beside the button rather than inside it, so no control is nested in another; the dropzone opens it by
          reference. Hidden, but still a form control with a name (#101). */}
      <input {...getInputProps()} aria-label="Choose a CSV file" />
    </>
  );
}
