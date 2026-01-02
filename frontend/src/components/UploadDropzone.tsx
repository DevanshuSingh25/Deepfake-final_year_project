import { useCallback } from 'react';
import { useDropzone, Accept } from 'react-dropzone';
import { Upload, File as FileIcon } from 'lucide-react';
import { cn } from '@/lib/utils';

interface UploadDropzoneProps {
  onFileSelect: (file: File) => void;
  accept: string;
  maxSize?: number;
  disabled?: boolean;
}

/**
 * Parse accept string into react-dropzone Accept format.
 * Handles both MIME types (audio/wav) and extensions (.wav)
 */
function parseAccept(acceptString: string): Accept {
  const items = acceptString.split(',').map(s => s.trim());
  const result: Accept = {};

  for (const item of items) {
    if (item.startsWith('.')) {
      // It's a file extension - map to appropriate MIME type
      const ext = item.toLowerCase();
      switch (ext) {
        case '.wav':
          result['audio/wav'] = result['audio/wav'] || [];
          result['audio/wav'].push('.wav');
          result['audio/x-wav'] = result['audio/x-wav'] || [];
          result['audio/x-wav'].push('.wav');
          break;
        case '.mp3':
          result['audio/mpeg'] = result['audio/mpeg'] || [];
          result['audio/mpeg'].push('.mp3');
          break;
        case '.m4a':
          result['audio/mp4'] = result['audio/mp4'] || [];
          result['audio/mp4'].push('.m4a');
          result['audio/x-m4a'] = result['audio/x-m4a'] || [];
          result['audio/x-m4a'].push('.m4a');
          break;
        case '.flac':
          result['audio/flac'] = result['audio/flac'] || [];
          result['audio/flac'].push('.flac');
          result['audio/x-flac'] = result['audio/x-flac'] || [];
          result['audio/x-flac'].push('.flac');
          break;
        case '.ogg':
          result['audio/ogg'] = result['audio/ogg'] || [];
          result['audio/ogg'].push('.ogg');
          break;
        case '.mp4':
          result['video/mp4'] = result['video/mp4'] || [];
          result['video/mp4'].push('.mp4');
          break;
        case '.mov':
          result['video/quicktime'] = result['video/quicktime'] || [];
          result['video/quicktime'].push('.mov');
          break;
        case '.mkv':
          result['video/x-matroska'] = result['video/x-matroska'] || [];
          result['video/x-matroska'].push('.mkv');
          break;
        case '.webm':
          result['video/webm'] = result['video/webm'] || [];
          result['video/webm'].push('.webm');
          break;
        default:
          // Unknown extension - add as-is with wildcard mime
          result['application/octet-stream'] = result['application/octet-stream'] || [];
          result['application/octet-stream'].push(ext);
      }
    } else if (item.includes('/')) {
      // It's a MIME type
      result[item] = result[item] || [];
    }
  }

  return result;
}

/**
 * Get display-friendly format string from accept prop
 */
function getDisplayFormats(acceptString: string): string {
  const items = acceptString.split(',').map(s => s.trim());
  const extensions = items
    .filter(item => item.startsWith('.'))
    .map(ext => ext.toUpperCase().slice(1));

  if (extensions.length > 0) {
    return extensions.join(', ');
  }

  // If no extensions, extract from MIME types
  return items
    .filter(item => item.includes('/'))
    .map(mime => mime.split('/')[1]?.toUpperCase())
    .filter(Boolean)
    .join(', ');
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
    accept: parseAccept(accept),
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
            Accepted formats: {getDisplayFormats(accept)}
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
