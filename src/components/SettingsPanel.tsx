import { Card, CardHeader, CardTitle, CardContent } from './Card';
import { Settings as SettingsIcon } from 'lucide-react';

export interface VideoSettings {
  frameSamplingRate: number;
  faceFocus: boolean;
}

export interface AudioSettings {
  chunkSize: string;
  spectrogramMode: boolean;
  noiseReduction: boolean;
}

interface VideoSettingsPanelProps {
  type: 'video';
  settings: VideoSettings;
  onChange: (settings: VideoSettings) => void;
}

interface AudioSettingsPanelProps {
  type: 'audio';
  settings: AudioSettings;
  onChange: (settings: AudioSettings) => void;
}

type SettingsPanelProps = VideoSettingsPanelProps | AudioSettingsPanelProps;

export function SettingsPanel({ type, settings, onChange }: SettingsPanelProps) {
  if (type === 'video') {
    const videoSettings = settings as VideoSettings;

    return (
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-lg">
            <SettingsIcon className="w-5 h-5" />
            Analysis Settings
          </CardTitle>
        </CardHeader>
        <CardContent className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-2">
              Frame Sampling Rate
            </label>
            <select
              value={videoSettings.frameSamplingRate}
              onChange={(e) =>
                onChange({ ...videoSettings, frameSamplingRate: Number(e.target.value) })
              }
              className="w-full px-3 py-2 rounded-lg bg-background border border-input focus:ring-2 focus:ring-ring focus:border-ring"
            >
              <option value="10">10 frames (84% accuracy)</option>
              <option value="20">20 frames (90% accuracy)</option>
              <option value="40">40 frames (95% accuracy)</option>
              <option value="60">60 frames (97% accuracy)</option>
              <option value="80">80 frames (97% accuracy)</option>
              <option value="100">100 frames (97% accuracy)</option>
            </select>
            <p className="text-xs text-muted-foreground mt-1">
              Higher frame counts provide better accuracy but take longer to process
            </p>
          </div>

          <div className="flex items-center justify-between">
            <label className="text-sm font-medium">Face Focus</label>
            <input
              type="checkbox"
              checked={videoSettings.faceFocus}
              onChange={(e) =>
                onChange({ ...videoSettings, faceFocus: e.target.checked })
              }
              className="w-4 h-4 rounded border-input focus:ring-2 focus:ring-ring"
            />
          </div>
        </CardContent>
      </Card>
    );
  }

  // Audio settings
  const audioSettings = settings as AudioSettings;

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-lg">
          <SettingsIcon className="w-5 h-5" />
          Analysis Settings
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div>
          <label className="block text-sm font-medium mb-2">Chunk Size</label>
          <select
            value={audioSettings.chunkSize}
            onChange={(e) =>
              onChange({ ...audioSettings, chunkSize: e.target.value })
            }
            className="w-full px-3 py-2 rounded-lg bg-background border border-input focus:ring-2 focus:ring-ring focus:border-ring"
          >
            <option value="auto">Auto</option>
            <option value="2s">2 seconds</option>
            <option value="3s">3 seconds</option>
            <option value="5s">5 seconds</option>
          </select>
        </div>

        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">Spectrogram Mode</label>
          <input
            type="checkbox"
            checked={audioSettings.spectrogramMode}
            onChange={(e) =>
              onChange({ ...audioSettings, spectrogramMode: e.target.checked })
            }
            className="w-4 h-4 rounded border-input focus:ring-2 focus:ring-ring"
          />
        </div>

        <div className="flex items-center justify-between">
          <label className="text-sm font-medium">Noise Reduction</label>
          <input
            type="checkbox"
            checked={audioSettings.noiseReduction}
            onChange={(e) =>
              onChange({ ...audioSettings, noiseReduction: e.target.checked })
            }
            className="w-4 h-4 rounded border-input focus:ring-2 focus:ring-ring"
          />
        </div>
      </CardContent>
    </Card>
  );
}
