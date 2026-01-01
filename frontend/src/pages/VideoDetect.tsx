import { useState } from 'react';
import { Video, RotateCcw, ShieldCheck, ShieldAlert, Loader2 } from 'lucide-react';
import { Button } from '@/components/Button';
import { UploadDropzone } from '@/components/UploadDropzone';
import { MediaPreviewVideo } from '@/components/MediaPreviewVideo';
import { SettingsPanel, VideoSettings } from '@/components/SettingsPanel';
import { Toast, ToastType } from '@/components/Toast';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/Card';
import { validateVideoFile, MAX_VIDEO_SIZE } from '@/lib/validators';
import { getVideoDuration } from '@/lib/utils';
import { storage } from '@/lib/storage';
import { VideoPredictionResponse } from '@/lib/api';

interface AnalysisResult {
  prediction: 'REAL' | 'FAKE';
  confidence: number;
  facesFound: number;
  totalFrames: number;
  frameImages: string[];
}

export default function VideoDetect() {
  const [file, setFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number>(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
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
    setResult(null);

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
    setResult(null);

    try {
      // Import the predictVideo function dynamically
      const { predictVideo } = await import('@/lib/api');

      const response: VideoPredictionResponse = await predictVideo({
        file,
        frameSamplingRate: settings.frameSamplingRate,
        faceFocus: settings.faceFocus,
      });

      setIsAnalyzing(false);

      // Store the analysis result
      setResult({
        prediction: response.prediction,
        confidence: response.confidence,
        facesFound: response.faces_found,
        totalFrames: response.total_frames_analyzed,
        frameImages: response.frame_images,
      });

      setToast({ type: 'success', message: 'Analysis complete!' });
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
    setResult(null);
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
                {isAnalyzing ? 'Analyzing...' : 'Analyze Video'}
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

            {/* Analysis Result Panel */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  {result ? (
                    result.prediction === 'REAL' ? (
                      <ShieldCheck className="w-5 h-5 text-green-500" />
                    ) : (
                      <ShieldAlert className="w-5 h-5 text-red-500" />
                    )
                  ) : (
                    <Video className="w-5 h-5" />
                  )}
                  Analysis Result
                </CardTitle>
              </CardHeader>
              <CardContent>
                {isAnalyzing ? (
                  <div className="flex flex-col items-center justify-center py-8 gap-4">
                    <Loader2 className="w-12 h-12 animate-spin text-primary" />
                    <p className="text-muted-foreground">Analyzing video frames...</p>
                    <p className="text-xs text-muted-foreground">This may take up to a minute</p>
                  </div>
                ) : result ? (
                  <div className="space-y-6">
                    {/* Prediction Result */}
                    <div className={`p-6 rounded-xl text-center ${result.prediction === 'REAL'
                        ? 'bg-green-500/10 border border-green-500/30'
                        : 'bg-red-500/10 border border-red-500/30'
                      }`}>
                      <div className={`text-4xl font-bold mb-2 ${result.prediction === 'REAL' ? 'text-green-500' : 'text-red-500'
                        }`}>
                        {result.prediction}
                      </div>
                      <div className="text-2xl font-semibold text-foreground">
                        {result.confidence.toFixed(1)}% Confidence
                      </div>
                    </div>

                    {/* Analysis Details */}
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div className="p-3 rounded-lg bg-muted/50">
                        <p className="text-muted-foreground">Faces Detected</p>
                        <p className="font-semibold text-lg">{result.facesFound}/{result.totalFrames}</p>
                      </div>
                      <div className="p-3 rounded-lg bg-muted/50">
                        <p className="text-muted-foreground">Frames Analyzed</p>
                        <p className="font-semibold text-lg">{result.totalFrames}</p>
                      </div>
                    </div>

                    {/* Sampled Frames */}
                    {result.frameImages.length > 0 && (
                      <div>
                        <h4 className="text-sm font-medium mb-3">Sampled Frames (Face Detection)</h4>
                        <div className="grid grid-cols-3 gap-2">
                          {result.frameImages.map((imgSrc, idx) => (
                            <div
                              key={idx}
                              className="aspect-square rounded-lg overflow-hidden bg-muted border"
                            >
                              <img
                                src={imgSrc}
                                alt={`Frame ${idx + 1}`}
                                className="w-full h-full object-cover"
                              />
                            </div>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <Video className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Upload a video and click "Analyze Video" to see results</p>
                  </div>
                )}
              </CardContent>
            </Card>
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
