import { useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { Upload, File as FileIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void;
  accept: string;
  maxSize?: number;
  disabled?: boolean;
}

export function UploadDropzone({
  onFileSelect,
  accept,
  maxSize,
  disabled,
}: UploadDropzoneProps) {
  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileSelect(acceptedFiles[0]);
      }
    },
    [onFileSelect]
  );

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: accept.split(',').reduce((acc, ext) => {
      acc[ext.trim()] = [];
      return acc;
    }, {} as Record<string, string[]>),
    maxSize,
    multiple: false,
    disabled,
  });

  return (
    <div
      {...getRootProps()}
      className={cn(
        'border-2 border-dashed rounded-2xl p-8 text-center cursor-pointer transition-all focus-ring',
        isDragActive
          ? 'border-primary bg-primary/5'
          : 'border-border hover:border-primary/50 hover:bg-muted/50',
        disabled && 'opacity-50 cursor-not-allowed'
      )}
    >
      <input {...getInputProps()} />
      
      <div className="flex flex-col items-center space-y-4">
        <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center">
          {isDragActive ? (
            <FileIcon className="w-8 h-8 text-primary animate-bounce" />
          ) : (
            <Upload className="w-8 h-8 text-primary" />
          )}
        </div>
        
        <div>
          <p className="text-lg font-medium mb-2">
            {isDragActive ? 'Drop file here' : 'Drag & drop or click to upload'}
          </p>
          <p className="text-sm text-muted-foreground">
            Accepted formats: {accept}
          </p>
          {maxSize && (
            <p className="text-xs text-muted-foreground mt-1">
              Maximum size: {(maxSize / (1024 * 1024)).toFixed(0)}MB
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
