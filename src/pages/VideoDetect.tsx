import { useState } from 'react';
import { Video, RotateCcw } from 'lucide-react';
import { Button } from '@/components/Button';
import { UploadDropzone } from '@/components/UploadDropzone';
import { MediaPreviewVideo } from '@/components/MediaPreviewVideo';
import { ResultPanel } from '@/components/ResultPanel';
import { SettingsPanel, VideoSettings } from '@/components/SettingsPanel';
import { Toast, ToastType } from '@/components/Toast';
import { validateVideoFile, MAX_VIDEO_SIZE } from '@/lib/validators';
import { getVideoDuration } from '@/lib/utils';
import { storage } from '@/lib/storage';

export default function VideoDetect() {
  const [file, setFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number>(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [hasResult, setHasResult] = useState(false);
  const [toast, setToast] = useState<{ type: ToastType; message: string } | null>(null);

  const [settings, setSettings] = useState<VideoSettings>({
    frameSamplingRate: 20,
    faceFocus: true,
  });

  const handleFileSelect = async (selectedFile: File) => {
    const validation = validateVideoFile(selectedFile);

    if (!validation.valid) {
      setToast({ type: 'error', message: validation.error! });
      return;
    }

    setFile(selectedFile);
    setHasResult(false);

    // Get video duration
    const dur = await getVideoDuration(selectedFile);
    setDuration(dur);

    // Add to recent uploads
    storage.addRecentUpload({
      filename: selectedFile.name,
      type: 'video',
      size: selectedFile.size,
    });

    setToast({ type: 'success', message: 'Video loaded successfully' });
  };

  const handleAnalyze = async () => {
    if (!file) {
      setToast({ type: 'warning', message: 'Please upload a video file first' });
      return;
    }

    setIsAnalyzing(true);

    try {
      // Import the predictVideo function dynamically
      const { predictVideo } = await import('@/lib/api');

      const response = await predictVideo({
        file,
        frameSamplingRate: settings.frameSamplingRate,
        faceFocus: settings.faceFocus,
      });

      setIsAnalyzing(false);
      setHasResult(true);

      // Store the actual prediction result
      const resultMessage = `Analysis complete: ${response.prediction} (${response.confidence.toFixed(1)}% confidence)`;
      setToast({ type: 'success', message: resultMessage });

      // You can store response.preprocessedImages and response.faceCroppedImages if needed
      // for displaying in the UI
    } catch (error) {
      setIsAnalyzing(false);
      const errorMessage = error instanceof Error ? error.message : 'Analysis failed';
      setToast({ type: 'error', message: errorMessage });
      console.error('Video analysis error:', error);
    }
  };

  const handleReset = () => {
    setFile(null);
    setDuration(0);
    setHasResult(false);
    setIsAnalyzing(false);
  };

  return (
    <div className="min-h-screen py-12 bg-gradient-to-b from-background to-muted/30">
      <div className="container mx-auto px-4">
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-4">
            <Video className="w-4 h-4 text-primary" />
            <span className="text-sm font-medium">Video Analysis</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Video Deepfake Detection
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload a video file to analyze for deepfake manipulation using ResNeXt + LSTM
          </p>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
          {/* Left Column */}
          <div className="space-y-6">
            <UploadDropzone
              onFileSelect={handleFileSelect}
              accept="video/mp4,video/quicktime,video/x-matroska,video/webm"
              maxSize={MAX_VIDEO_SIZE}
              disabled={isAnalyzing}
            />

            <SettingsPanel
              type="video"
              settings={settings}
              onChange={setSettings}
            />

            <div className="flex gap-4">
              <Button
                onClick={handleAnalyze}
                loading={isAnalyzing}
                disabled={!file || isAnalyzing}
                className="flex-1"
                variant="primary"
                size="lg"
              >
                Analyze Video
              </Button>
              <Button
                onClick={handleReset}
                disabled={isAnalyzing}
                variant="outline"
                size="lg"
              >
                <RotateCcw className="w-5 h-5" />
              </Button>
            </div>
          </div>

          {/* Right Column */}
          <div className="space-y-6">
            {file && <MediaPreviewVideo file={file} duration={duration} />}
            <ResultPanel hasResult={hasResult} />
          </div>
        </div>
      </div>

      {/* Toast Notifications */}
      {toast && (
        <Toast
          type={toast.type}
          message={toast.message}
          onClose={() => setToast(null)}
        />
      )}
    </div>
  );
}
