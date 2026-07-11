import { useCallback, useState } from "react";
import { useDropzone } from "react-dropzone";
import { useTranslation } from "react-i18next";
import { Fingerprint, UploadCloud } from "lucide-react";

export default function UploadDropzone({ onFileSelected, disabled }) {
  const { t } = useTranslation();
  const [preview, setPreview] = useState(null);

  const onDrop = useCallback(
    (acceptedFiles) => {
      const file = acceptedFiles[0];
      if (!file) return;
      setPreview(URL.createObjectURL(file));
      onFileSelected(file);
    },
    [onFileSelected]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    disabled,
    accept: { "image/png": [".png"], "image/jpeg": [".jpg", ".jpeg"], "image/bmp": [".bmp"] },
    maxFiles: 1,
    maxSize: 16 * 1024 * 1024,
  });

  return (
    <div
      {...getRootProps()}
      className={`relative flex min-h-[260px] cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-2 border-dashed bg-grid bg-grid p-8 text-center transition-colors ${
        isDragActive ? "border-signal bg-signal/5" : "border-line hover:border-signal/50"
      } ${disabled ? "cursor-not-allowed opacity-60" : ""}`}
    >
      <input {...getInputProps()} />
      {preview ? (
        <img src={preview} alt="Fingerprint preview" className="max-h-48 rounded border border-line object-contain" />
      ) : (
        <>
          <div className="rounded-full border border-line bg-panel-raised p-4">
            {isDragActive ? <UploadCloud size={28} className="text-signal" /> : <Fingerprint size={28} className="text-muted" />}
          </div>
          <p className="font-mono text-sm text-primary">{isDragActive ? t("upload.dragActive") : t("upload.dragInactive")}</p>
          <p className="text-xs text-muted">{t("upload.supportedFormats")}</p>
        </>
      )}
    </div>
  );
}
