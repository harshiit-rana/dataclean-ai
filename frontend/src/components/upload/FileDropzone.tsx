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
    <div
      {...getRootProps()}
      className={cn(
        "flex cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-border p-10 text-center transition-colors",
        isDragActive && "border-primary bg-muted"
      )}
    >
      {/* Hidden by the dropzone, but still a form control: it needs a name a screen reader can read (#101). */}
      <input {...getInputProps()} aria-label="Choose a CSV file" />
      {selectedFile ? (
        <p className="text-sm font-medium">{selectedFile.name}</p>
      ) : (
        <>
          <p className="text-sm font-medium">Drop a CSV file here, or click to browse</p>
          <p className="mt-1 text-xs text-muted-foreground">CSV only, up to 50MB</p>
        </>
      )}
    </div>
  );
}
