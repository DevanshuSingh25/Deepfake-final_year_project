const THEME_KEY = 'dg-theme';
const RECENT_UPLOADS_KEY = 'dg-recent-uploads';

export type Theme = 'light' | 'dark';

export interface RecentUpload {
  id: string;
  filename: string;
  type: 'video' | 'audio';
  timestamp: number;
  size: number;
}

export const storage = {
  // Theme management
  getTheme(): Theme {
    const stored = localStorage.getItem(THEME_KEY);
    if (stored === 'light' || stored === 'dark') return stored;
    
    // Default to system preference
    if (window.matchMedia('(prefers-color-scheme: dark)').matches) {
      return 'dark';
    }
    return 'light';
  },

  setTheme(theme: Theme): void {
    localStorage.setItem(THEME_KEY, theme);
    
    if (theme === 'dark') {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  },

  // Recent uploads management
  getRecentUploads(): RecentUpload[] {
    const stored = localStorage.getItem(RECENT_UPLOADS_KEY);
    if (!stored) return [];
    
    try {
      return JSON.parse(stored);
    } catch {
      return [];
    }
  },

  addRecentUpload(upload: Omit<RecentUpload, 'id' | 'timestamp'>): void {
    const recent = this.getRecentUploads();
    const newUpload: RecentUpload = {
      ...upload,
      id: Date.now().toString(),
      timestamp: Date.now(),
    };
    
    // Keep only last 10 uploads
    const updated = [newUpload, ...recent].slice(0, 10);
    localStorage.setItem(RECENT_UPLOADS_KEY, JSON.stringify(updated));
  },

  clearRecentUploads(): void {
    localStorage.removeItem(RECENT_UPLOADS_KEY);
  },
};

