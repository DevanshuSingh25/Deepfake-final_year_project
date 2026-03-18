import { useState } from 'react';
import { Music, RotateCcw, ShieldCheck, ShieldAlert, Loader2 } from 'lucide-react';
import { Button } from '@/components/Button';
import { UploadDropzone } from '@/components/UploadDropzone';
import { MediaPreviewAudio } from '@/components/MediaPreviewAudio';
import { Toast, ToastType } from '@/components/Toast';
import { Card, CardHeader, CardTitle, CardContent } from '@/components/Card';
import { validateAudioFile, MAX_AUDIO_SIZE } from '@/lib/validators';
import { getAudioDuration } from '@/lib/utils';
import { storage } from '@/lib/storage';
import { AudioPredictionResponse } from '@/lib/api';

interface AnalysisResult {
  prediction: 'REAL' | 'FAKE';
  confidence: number;
  model: string;
  allScores: {
    real: number;
    fake: number;
  };
}

export default function AudioDetect() {
  const [file, setFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number>(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [toast, setToast] = useState<{ type: ToastType; message: string } | null>(null);

  const handleFileSelect = async (selectedFile: File) => {
    const validation = validateAudioFile(selectedFile);

    if (!validation.valid) {
      setToast({ type: 'error', message: validation.error! });
      return;
    }

    // 🔥 TRACK AUDIO UPLOAD
    window.dataLayer = window.dataLayer || [];
    window.dataLayer.push({
      event: "audio_upload",
      file_name: selectedFile.name,
      file_size: (selectedFile.size / 1024 / 1024).toFixed(2) + "MB",
      file_type: selectedFile.type
    });

    setFile(selectedFile);
    setResult(null);

    const dur = await getAudioDuration(selectedFile);
    setDuration(dur);

    storage.addRecentUpload({
      filename: selectedFile.name,
      type: 'audio',
      size: selectedFile.size,
    });

    setToast({ type: 'success', message: 'Audio loaded successfully' });
  };

  const handleAnalyze = async () => {
    if (!file) {
      setToast({ type: 'warning', message: 'Please upload an audio file first' });
      return;
    }

    // 🔥 TRACK ANALYZE BUTTON CLICK
    window.dataLayer = window.dataLayer || [];
    window.dataLayer.push({
      event: "analyze_audio_click"
    });

    setIsAnalyzing(true);
    setResult(null);

    try {
      const { predictAudio } = await import('@/lib/api');

      const response: AudioPredictionResponse = await predictAudio({ file });

      setIsAnalyzing(false);

      // 🔥 TRACK ANALYSIS COMPLETE WITH FULL DATA
      window.dataLayer = window.dataLayer || [];
      window.dataLayer.push({
        event: "analysis_complete",
        result: response.prediction,
        confidence: response.confidence,
        file_name: file?.name,
        file_size: (file?.size / 1024 / 1024).toFixed(2) + "MB",
        file_type: file?.type
      });

      setResult({
        prediction: response.prediction,
        confidence: response.confidence,
        model: response.model,
        allScores: response.all_scores,
      });

      setToast({ type: 'success', message: 'Analysis complete!' });
    } catch (error) {
      setIsAnalyzing(false);
      const errorMessage = error instanceof Error ? error.message : 'Analysis failed';
      setToast({ type: 'error', message: errorMessage });

      // 🔥 OPTIONAL: TRACK FAILURE
      window.dataLayer = window.dataLayer || [];
      window.dataLayer.push({
        event: "analysis_failed",
        error: errorMessage
      });

      console.error('Audio analysis error:', error);
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

        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-4">
            <Music className="w-4 h-4 text-secondary" />
            <span className="text-sm font-medium">Audio Analysis</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Audio Deepfake Detection
          </h1>
        </div>

        <div className="grid lg:grid-cols-2 gap-8 max-w-7xl mx-auto">

          <div className="space-y-6">
            <UploadDropzone
              onFileSelect={handleFileSelect}
              accept="audio/wav,audio/mpeg,audio/mp3,audio/mp4,audio/flac,audio/x-flac,.wav,.mp3,.m4a,.flac"
              maxSize={MAX_AUDIO_SIZE}
              disabled={isAnalyzing}
            />

            <div className="flex gap-4">
              <Button
                onClick={handleAnalyze}
                loading={isAnalyzing}
                disabled={!file || isAnalyzing}
                className="flex-1"
                variant="secondary"
                size="lg"
              >
                {isAnalyzing ? 'Analyzing...' : 'Analyze Audio'}
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

          <div className="space-y-6">
            {file && <MediaPreviewAudio file={file} duration={duration} />}

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
                    <Music className="w-5 h-5" />
                  )}
                  Analysis Result
                </CardTitle>
              </CardHeader>

              <CardContent>
                {isAnalyzing ? (
                  <div className="flex flex-col items-center justify-center py-8 gap-4">
                    <Loader2 className="w-12 h-12 animate-spin text-secondary" />
                    <p className="text-muted-foreground">Analyzing audio...</p>
                  </div>
                ) : result ? (
                  <div className="space-y-6">
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
                  </div>
                ) : (
                  <div className="text-center py-8 text-muted-foreground">
                    <Music className="w-12 h-12 mx-auto mb-4 opacity-50" />
                    <p>Upload an audio file and click "Analyze Audio" to see results</p>
                  </div>
                )}
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

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
