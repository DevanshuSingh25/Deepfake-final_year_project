import { useState } from 'react';
import { Music, RotateCcw } from 'lucide-react';
import { Button } from '@/components/Button';
import { UploadDropzone } from '@/components/UploadDropzone';
import { MediaPreviewAudio } from '@/components/MediaPreviewAudio';
import { ResultPanel } from '@/components/ResultPanel';
import { SettingsPanel, AudioSettings } from '@/components/SettingsPanel';
import { Toast, ToastType } from '@/components/Toast';
import { validateAudioFile, MAX_AUDIO_SIZE } from '@/lib/validators';
import { getAudioDuration } from '@/lib/utils';
import { storage } from '@/lib/storage';

export default function AudioDetect() {
  const [file, setFile] = useState<File | null>(null);
  const [duration, setDuration] = useState<number>(0);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [hasResult, setHasResult] = useState(false);
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
    setHasResult(false);
    
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

  const handleAnalyze = () => {
    if (!file) {
      setToast({ type: 'warning', message: 'Please upload an audio file first' });
      return;
    }

    setIsAnalyzing(true);
    
    // Simulate analysis (UI only)
    setTimeout(() => {
      setIsAnalyzing(false);
      setHasResult(true);
      setToast({ type: 'success', message: 'Analysis complete' });
    }, 3000);
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
            <Music className="w-4 h-4 text-secondary" />
            <span className="text-sm font-medium">Audio Analysis</span>
          </div>
          <h1 className="text-4xl md:text-5xl font-bold mb-4">
            Audio Deepfake Detection
          </h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Upload an audio file to detect voice cloning and synthetic audio using ResNeXt + LSTM
          </p>
        </div>

        {/* Main Content */}
        <div className="grid lg:grid-cols-2 gap-8 max-w-7xl mx-auto">
          {/* Left Column */}
          <div className="space-y-6">
            <UploadDropzone
              onFileSelect={handleFileSelect}
              accept=".wav,.mp3,.m4a,.flac"
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
                Analyze Audio
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
