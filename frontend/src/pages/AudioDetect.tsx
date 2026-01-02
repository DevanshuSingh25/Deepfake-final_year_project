import { useState } from 'react';
import { Music, RotateCcw, ShieldCheck, ShieldAlert, Loader2 } from 'lucide-react';
import { Button } from '@/components/Button';
import { UploadDropzone } from '@/components/UploadDropzone';
import { MediaPreviewAudio } from '@/components/MediaPreviewAudio';
import { SettingsPanel, AudioSettings } from '@/components/SettingsPanel';
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

  const [settings, setSettings] = useState<AudioSettings>({
    chunkSize: 'auto',
    spectrogramMode: true,
    noiseReduction: false,
  });

  const handleFileSelect = async (selectedFile: File) => {
    const validation = validateAudioFile(selectedFile);

    if (!validation.valid) {
      setToast({ type: 'error', message: validation.error! });
      return;
    }

    setFile(selectedFile);
    setResult(null);

    // Get audio duration
    const dur = await getAudioDuration(selectedFile);
    setDuration(dur);

    // Add to recent uploads
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

    setIsAnalyzing(true);
    setResult(null);

    try {
      // Import the predictAudio function dynamically
      const { predictAudio } = await import('@/lib/api');

      const response: AudioPredictionResponse = await predictAudio({ file });

      setIsAnalyzing(false);

      // Store the analysis result
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
        {/* Header */}
        <div className="text-center mb-12">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-4">
            <Music className="w-4 h-4 text-secondary" />
            <span className="text-sm font-medium">Audio Analysis</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Audio Deepfake Detection
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload an audio file to detect voice cloning and synthetic audio using Wav2Vec2
          </p>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
          {/* Left Column */}
          <div className="space-y-6">
            <UploadDropzone
              onFileSelect={handleFileSelect}
              accept="audio/wav,audio/mpeg,audio/mp3,audio/mp4,audio/flac,audio/x-flac,.wav,.mp3,.m4a,.flac"
              maxSize={MAX_AUDIO_SIZE}
              disabled={isAnalyzing}
            />

            <SettingsPanel
              type="audio"
              settings={settings}
              onChange={setSettings}
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

          {/* Right Column */}
          <div className="space-y-6">
            {file && <MediaPreviewAudio file={file} duration={duration} />}

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
                    <p className="text-xs text-muted-foreground">First analysis may take longer (model download)</p>
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

                    {/* Score Breakdown */}
                    <div className="space-y-3">
                      <h4 className="text-sm font-medium text-muted-foreground">Score Breakdown</h4>

                      {/* Real Score Bar */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-green-500 font-medium">Real</span>
                          <span>{result.allScores.real.toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full bg-green-500 transition-all duration-500"
                            style={{ width: `${result.allScores.real}%` }}
                          />
                        </div>
                      </div>

                      {/* Fake Score Bar */}
                      <div className="space-y-1">
                        <div className="flex justify-between text-sm">
                          <span className="text-red-500 font-medium">Fake</span>
                          <span>{result.allScores.fake.toFixed(1)}%</span>
                        </div>
                        <div className="w-full h-2 bg-muted rounded-full overflow-hidden">
                          <div
                            className="h-full bg-red-500 transition-all duration-500"
                            style={{ width: `${result.allScores.fake}%` }}
                          />
                        </div>
                      </div>
                    </div>

                    {/* Model Info */}
                    <div className="p-3 rounded-lg bg-muted/50 text-center">
                      <p className="text-xs text-muted-foreground">Model</p>
                      <p className="text-sm font-medium truncate">{result.model}</p>
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
