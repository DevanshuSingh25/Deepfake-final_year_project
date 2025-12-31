import { useEffect, useRef, useState } from 'react';
import WaveSurfer from 'wavesurfer.js';
import SpectrogramPlugin from 'wavesurfer.js/dist/plugins/spectrogram';
import { Music, Play, Pause, Loader2 } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './Card';
import { Button } from './Button';
import { formatBytes, formatDuration } from '@/lib/utils';

interface MediaPreviewAudioProps {
  file: File | null;
  duration?: number;
}

export function MediaPreviewAudio({ file, duration }: MediaPreviewAudioProps) {
  const waveformRef = useRef<HTMLDivElement>(null);
  const spectrogramRef = useRef<HTMLDivElement>(null);
  const wavesurferRef = useRef<WaveSurfer | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (file && waveformRef.current && spectrogramRef.current) {
      setLoading(true);

      // Initialize WaveSurfer
      const wavesurfer = WaveSurfer.create({
        container: waveformRef.current,
        waveColor: 'hsl(var(--primary))',
        progressColor: 'hsl(var(--secondary))',
        height: 80,
        normalize: true,
        barWidth: 2,
        barGap: 1,
        barRadius: 2,
      });

      // Add spectrogram plugin
      wavesurfer.registerPlugin(
        SpectrogramPlugin.create({
          container: spectrogramRef.current!,
          labels: true,
          height: 128,
        })
      );

      // Load audio file
      const url = URL.createObjectURL(file);
      wavesurfer.load(url);

      wavesurfer.on('ready', () => {
        setLoading(false);
        URL.revokeObjectURL(url);
      });

      wavesurfer.on('play', () => setIsPlaying(true));
      wavesurfer.on('pause', () => setIsPlaying(false));

      wavesurferRef.current = wavesurfer;

      return () => {
        wavesurfer.destroy();
        wavesurferRef.current = null;
      };
    }
  }, [file]);

  const handlePlayPause = () => {
    if (wavesurferRef.current) {
      wavesurferRef.current.playPause();
    }
  };

  if (!file) {
    return null;
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2">
          <Music className="w-5 h-5" />
          Audio Preview
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Waveform */}
        <div>
          <div className="flex items-center justify-between mb-2">
            <h4 className="text-sm font-medium">Waveform</h4>
            <Button
              size="sm"
              variant="outline"
              onClick={handlePlayPause}
              disabled={loading}
            >
              {loading ? (
                <Loader2 className="w-4 h-4 animate-spin" />
              ) : isPlaying ? (
                <Pause className="w-4 h-4" />
              ) : (
                <Play className="w-4 h-4" />
              )}
            </Button>
          </div>
          <div
            ref={waveformRef}
            className="rounded-lg overflow-hidden bg-muted border"
          />
        </div>

        {/* Spectrogram */}
        <div>
          <h4 className="text-sm font-medium mb-2">Spectrogram</h4>
          <div
            ref={spectrogramRef}
            className="rounded-lg overflow-hidden bg-muted border"
          />
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
      </CardContent>
    </Card>
  );
}
