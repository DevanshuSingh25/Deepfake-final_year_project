import { useEffect, useState } from 'react';
import { Film, Loader2 } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './Card';
import { extractVideoFrames, formatBytes, formatDuration } from '@/lib/utils';

interface MediaPreviewVideoProps {
  file: File | null;
  duration?: number;
}

export function MediaPreviewVideo({ file, duration }: MediaPreviewVideoProps) {
  const [url, setUrl] = useState<string>('');
  const [frames, setFrames] = useState<string[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (file) {
      const objectUrl = URL.createObjectURL(file);
      setUrl(objectUrl);
      
      // Extract frames
      setLoading(true);
      extractVideoFrames(file, 6)
        .then(setFrames)
        .finally(() => setLoading(false));

      return () => {
        URL.revokeObjectURL(objectUrl);
      };
    } else {
      setUrl('');
      setFrames([]);
    }
  }, [file]);

  if (!file) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Film className="w-5 h-5" />
          Video Preview
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Video Player */}
        <div className="aspect-video bg-black rounded-xl overflow-hidden">
          <video
            src={url}
            controls
            className="w-full h-full"
            controlsList="nodownload"
          >
            Your browser does not support the video tag.
          </video>
        </div>

        {/* File Details */}
        <div className="grid grid-cols-2 gap-4 text-sm">
          <div>
            <p className="text-muted-foreground">Filename</p>
            <p className="font-medium truncate">{file.name}</p>
          </div>
          <div>
            <p className="text-muted-foreground">Size</p>
            <p className="font-medium">{formatBytes(file.size)}</p>
          </div>
          {duration && (
            <div className="col-span-2">
              <p className="text-muted-foreground">Duration</p>
              <p className="font-medium">{formatDuration(duration)}</p>
            </div>
          )}
        </div>

        {/* Frame Thumbnails */}
        <div>
          <h4 className="text-sm font-medium mb-3 flex items-center gap-2">
            Sampled Frames
            {loading && <Loader2 className="w-4 h-4 animate-spin" />}
          </h4>
          <div className="grid grid-cols-3 gap-2">
            {frames.map((frame, idx) => (
              <div
                key={idx}
                className="aspect-video rounded-lg overflow-hidden bg-muted border"
              >
                <img
                  src={frame}
                  alt={`Frame ${idx + 1}`}
                  className="w-full h-full object-cover"
                />
              </div>
            ))}
            {loading && frames.length === 0 && (
              <>
                {[...Array(6)].map((_, idx) => (
                  <div
                    key={idx}
                    className="aspect-video rounded-lg bg-muted animate-pulse"
                  />
                ))}
              </>
            )}
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
