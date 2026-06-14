"use client";

import React, { useState, useRef, DragEvent, ChangeEvent } from "react";
import { UploadCloud, FileImage, X, AlertTriangle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Button } from "@/components/ui/button";

interface UploadAreaProps {
  onFileSelected: (file: File) => void;
  disabled?: boolean;
}

export function UploadArea({ onFileSelected, disabled = false }: UploadAreaProps) {
  const [isDragActive, setIsDragActive] = useState(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  
  const inputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (disabled) return;
    
    if (e.type === "dragenter" || e.type === "dragover") {
      setIsDragActive(true);
    } else if (e.type === "dragleave") {
      setIsDragActive(false);
    }
  };

  const processFile = (file: File) => {
    setError(null);
    
    // Validate file type
    const validTypes = ["image/jpeg", "image/jpg", "image/png"];
    if (!validTypes.includes(file.type) && !file.name.toLowerCase().endsWith(".jpg") && !file.name.toLowerCase().endsWith(".jpeg") && !file.name.toLowerCase().endsWith(".png")) {
      setError("Supported file formats are JPEG, JPG, and PNG only.");
      return;
    }
    
    // File size limit (e.g. 50MB for high-res drone photography)
    if (file.size > 50 * 1024 * 1024) {
      setError("File size exceeds 50MB limit.");
      return;
    }

    setSelectedFile(file);
    onFileSelected(file);
    
    // Create image preview
    const reader = new FileReader();
    reader.onloadend = () => {
      setPreviewUrl(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragActive(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      processFile(e.dataTransfer.files[0]);
    }
  };

  const handleChange = (e: ChangeEvent<HTMLInputElement>) => {
    e.preventDefault();
    if (disabled) return;

    if (e.target.files && e.target.files[0]) {
      processFile(e.target.files[0]);
    }
  };

  const onButtonClick = () => {
    if (disabled) return;
    inputRef.current?.click();
  };

  const handleClear = (e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedFile(null);
    setPreviewUrl(null);
    setError(null);
    if (inputRef.current) {
      inputRef.current.value = "";
    }
  };

  const formatFileSize = (bytes: number): string => {
    if (bytes === 0) return "0 Bytes";
    const k = 1024;
    const sizes = ["Bytes", "KB", "MB"];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i];
  };

  return (
    <div className="w-full">
      <input
        ref={inputRef}
        type="file"
        className="hidden"
        accept=".jpg,.jpeg,.png,image/jpeg,image/png"
        onChange={handleChange}
        disabled={disabled}
      />

      <AnimatePresence mode="wait">
        {!selectedFile ? (
          <motion.div
            key="dropzone"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            onDragEnter={handleDrag}
            onDragOver={handleDrag}
            onDragLeave={handleDrag}
            onDrop={handleDrop}
            onClick={onButtonClick}
            className={`relative flex flex-col items-center justify-center min-h-[260px] border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-all duration-300 ${
              isDragActive 
                ? "border-primary bg-primary/5 scale-[0.99] shadow-lg shadow-primary/5" 
                : "border-border/80 hover:border-primary/50 hover:bg-muted/30"
            } ${disabled ? "opacity-50 cursor-not-allowed" : ""}`}
          >
            <div className="flex flex-col items-center gap-4">
              <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-muted/60 text-muted-foreground border border-border/40 group-hover:scale-105 transition-transform duration-300">
                <UploadCloud className="h-7 w-7 text-primary" />
              </div>
              
              <div className="flex flex-col gap-1.5">
                <h3 className="font-semibold text-base tracking-tight">
                  Upload high-resolution aerial image
                </h3>
                <p className="text-xs text-muted-foreground max-w-[280px] mx-auto leading-relaxed">
                  Drag and drop files here, or click to browse. Supports JPG, JPEG, and PNG (up to 50MB).
                </p>
              </div>

              {error && (
                <div className="flex items-center gap-1.5 rounded-lg bg-destructive/10 px-3 py-1.5 text-xs font-medium text-destructive border border-destructive/20 animate-shake">
                  <AlertTriangle className="h-3.5 w-3.5" />
                  <span>{error}</span>
                </div>
              )}
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="preview"
            initial={{ opacity: 0, scale: 0.96 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.96 }}
            transition={{ type: "spring", damping: 20, stiffness: 200 }}
            className="border border-border/60 rounded-2xl overflow-hidden bg-card shadow-sm"
          >
            {/* Top Bar info */}
            <div className="flex items-center justify-between border-b border-border/40 px-4 py-3 bg-muted/30">
              <div className="flex items-center gap-2.5 min-w-0">
                <FileImage className="h-4 w-4 text-primary shrink-0" />
                <div className="flex flex-col min-w-0">
                  <span className="text-xs font-semibold truncate max-w-[200px] sm:max-w-xs text-card-foreground">
                    {selectedFile.name}
                  </span>
                  <span className="text-[10px] text-muted-foreground leading-none">
                    {formatFileSize(selectedFile.size)}
                  </span>
                </div>
              </div>

              <Button
                variant="ghost"
                size="icon"
                onClick={handleClear}
                className="h-8 w-8 rounded-lg hover:bg-muted text-muted-foreground hover:text-destructive transition-colors"
                disabled={disabled}
              >
                <X className="h-4 w-4" />
              </Button>
            </div>

            {/* Thumbnail preview */}
            {previewUrl && (
              <div className="relative aspect-[16/9] w-full bg-slate-950/20 dark:bg-slate-950/40 flex items-center justify-center p-2 group">
                <img
                  src={previewUrl}
                  alt="Aerial preview"
                  className="max-h-full max-w-full object-contain rounded-lg shadow-sm border border-border/30"
                />
                <div className="absolute inset-0 bg-background/20 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center backdrop-blur-[1px]">
                  <Button 
                    variant="secondary" 
                    size="sm" 
                    onClick={onButtonClick}
                    className="shadow-md"
                    disabled={disabled}
                  >
                    Replace Image
                  </Button>
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
