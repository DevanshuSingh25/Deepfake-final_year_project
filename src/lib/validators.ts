export const MAX_VIDEO_SIZE = 1024 * 1024 * 1024; // 1GB
export const MAX_AUDIO_SIZE = 100 * 1024 * 1024; // 100MB

export const VALID_VIDEO_TYPES = [
  'video/mp4',
  'video/quicktime',
  'video/x-matroska',
  'video/webm',
];

export const VALID_AUDIO_TYPES = [
  'audio/wav',
  'audio/mpeg',
  'audio/mp4',
  'audio/flac',
  'audio/x-flac',
];

export const VALID_VIDEO_EXTENSIONS = ['.mp4', '.mov', '.mkv', '.webm'];
export const VALID_AUDIO_EXTENSIONS = ['.wav', '.mp3', '.m4a', '.flac'];

export interface ValidationResult {
  valid: boolean;
  error?: string;
}

export function validateVideoFile(file: File): ValidationResult {
  if (!file) {
    return { valid: false, error: 'No file provided' };
  }

  // Check file size
  if (file.size > MAX_VIDEO_SIZE) {
    return {
      valid: false,
      error: `File size exceeds maximum of ${formatBytes(MAX_VIDEO_SIZE)}`,
    };
  }

  // Check file type
  const hasValidType = VALID_VIDEO_TYPES.includes(file.type);
  const hasValidExtension = VALID_VIDEO_EXTENSIONS.some((ext) =>
    file.name.toLowerCase().endsWith(ext)
  );

  if (!hasValidType && !hasValidExtension) {
    return {
      valid: false,
      error: `Invalid file type. Accepted formats: ${VALID_VIDEO_EXTENSIONS.join(', ')}`,
    };
  }

  return { valid: true };
}

export function validateAudioFile(file: File): ValidationResult {
  if (!file) {
    return { valid: false, error: 'No file provided' };
  }

  // Check file size
  if (file.size > MAX_AUDIO_SIZE) {
    return {
      valid: false,
      error: `File size exceeds maximum of ${formatBytes(MAX_AUDIO_SIZE)}`,
    };
  }

  // Check file type
  const hasValidType = VALID_AUDIO_TYPES.includes(file.type);
  const hasValidExtension = VALID_AUDIO_EXTENSIONS.some((ext) =>
    file.name.toLowerCase().endsWith(ext)
  );

  if (!hasValidType && !hasValidExtension) {
    return {
      valid: false,
      error: `Invalid file type. Accepted formats: ${VALID_AUDIO_EXTENSIONS.join(', ')}`,
    };
  }

  return { valid: true };
}

export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}
