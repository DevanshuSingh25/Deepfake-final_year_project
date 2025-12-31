# Deepfake Guard

> **AI-Powered Deepfake Detection for Video & Audio**

Advanced deepfake detection system using ResNeXt + LSTM neural networks. Full-stack application with SQLite authentication, built with React, TypeScript, Express, and Tailwind CSS.

![Deepfake Guard](https://lovable.dev/opengraph-image-p98pqg.png)

## 🚀 Features

### Core Capabilities
- **Video Deepfake Detection** - Analyze video files for face manipulation and synthetic content
- **Audio Deepfake Detection** - Detect voice cloning and synthetic audio generation
- **Frame Extraction** - Automatic sampling and thumbnail generation from video
- **Waveform & Spectrogram** - Real-time audio visualization with WaveSurfer.js
- **Analysis Settings** - Configurable frame rates and chunk sizes
- **User Authentication** - SQLite-based registration, login, and session management

### UI/UX Highlights
- **Glassmorphism Design** - Modern glass-card effects with backdrop blur
- **Animated Gradients** - Smooth gradient animations throughout
- **Dark Mode** - Persistent theme toggle with localStorage
- **Drag & Drop Upload** - react-dropzone integration for file uploads
- **Real-time Preview** - Video player and audio waveform/spectrogram visualization
- **Empty States** - Clear guidance when no data is present (no fake metrics)
- **Toast Notifications** - User feedback for all actions
- **Fully Responsive** - Mobile-first design with breakpoint optimization

### Technical Features
- **Code Splitting** - Lazy-loaded routes for optimal performance
- **Type Safety** - Full TypeScript coverage (frontend & backend)
- **Authentication** - JWT tokens with HTTP-only cookies
- **Accessibility** - ARIA labels, keyboard navigation, focus management
- **SEO Optimized** - Semantic HTML, meta tags, Open Graph support
- **Design System** - Centralized tokens in Tailwind config and CSS
- **Validation** - Client-side file validation & zod schema validation on backend

## 🎨 Design System

### Color Palette
- **Primary**: Indigo (HSL: 239, 84%, 67%)
- **Secondary**: Cyan (HSL: 189, 94%, 43%)
- **Base**: Slate/Ink tones for backgrounds

### Key Utilities
```css
/* Animated gradient */
.bg-gradient-hero

/* Glassmorphism */
.glass-card
.glass-card-strong

/* Animations */
.animate-gradient-shift
.animate-fade-in
.animate-scale-in
```

## 📁 Project Structure

```
src/
├── components/          # Reusable UI components
│   ├── Navbar.tsx       # Navigation with auth state
│   ├── Footer.tsx
│   ├── ThemeToggle.tsx
│   ├── Button.tsx
│   ├── Card.tsx
│   ├── Tabs.tsx
│   ├── UploadDropzone.tsx
│   ├── MediaPreviewVideo.tsx
│   ├── MediaPreviewAudio.tsx
│   ├── ResultPanel.tsx
│   ├── SettingsPanel.tsx  # Simplified settings
│   ├── Toast.tsx
│   ├── Modal.tsx
│   └── EmptyState.tsx
├── context/            # React context providers
│   └── AuthContext.tsx  # Auth state management
├── layouts/            # Layout components
│   └── AppLayout.tsx
├── pages/              # Route pages (lazy-loaded)
│   ├── Home.tsx
│   ├── VideoDetect.tsx  # Simplified settings
│   ├── AudioDetect.tsx  # Simplified settings
│   ├── Accounts.tsx     # Auth integration
│   ├── Contact.tsx      # Simplified contact page
│   └── NotFound.tsx
├── lib/                # Utilities and helpers
│   ├── api.ts          # API client for backend
│   ├── storage.ts      # Theme & upload history
│   ├── validators.ts   # File validation
│   └── utils.ts        # Formatting & frame extraction
├── router.tsx          # Route configuration
├── App.tsx             # Root component with AuthProvider
├── main.tsx            # Entry point
└── index.css           # Design system tokens

server/
├── index.ts            # Express server
├── auth.ts             # Authentication logic
├── db.ts               # SQLite database setup
├── env.ts              # Environment config
├── types.ts            # TypeScript types
├── package.json        # Server dependencies
└── tsconfig.json       # Server TypeScript config
```

## 🛠️ Tech Stack

### Frontend
- **React 18** - UI framework
- **TypeScript** - Type safety
- **Vite** - Build tool & dev server
- **Tailwind CSS** - Utility-first styling
- **react-router-dom** - Client-side routing with code splitting
- **@headlessui/react** - Accessible UI primitives (Tabs, Dialog)
- **react-dropzone** - Drag & drop file uploads
- **wavesurfer.js** - Audio waveform & spectrogram visualization
- **lucide-react** - Icon library

### Backend
- **Node.js + Express** - REST API server
- **TypeScript** - Type safety
- **SQLite (better-sqlite3)** - Embedded database
- **JWT (jsonwebtoken)** - Authentication tokens
- **bcryptjs** - Password hashing
- **zod** - Schema validation
- **CORS** - Cross-origin requests
- **cookie-parser** - Cookie handling

## 🚦 Getting Started

### Prerequisites
- Node.js 18+ and npm/pnpm/bun

### Installation

```bash
# Clone the repository
git clone <YOUR_GIT_URL>
cd deepfake-guard

# Install frontend dependencies
npm install

# Install server dependencies
cd server
npm install
cd ..

# Create .env file with JWT secret
echo "JWT_SECRET=your_secret_key_here_min_32_chars" > .env
echo "NODE_ENV=development" >> .env
```

### Running the Application

**IMPORTANT**: You need to manually add these scripts to `package.json` (the file is read-only in this environment):

```json
"scripts": {
  "dev": "vite",
  "server:dev": "cd server && npm install && npm run dev",
  "dev:all": "concurrently -k \"npm run dev\" \"npm run server:dev\""
}
```

Then run both frontend and backend:

```bash
npm run dev:all
```

Or run them separately:

```bash
# Terminal 1 - Frontend (port 8080)
npm run dev

# Terminal 2 - Backend (port 3001)
npm run server:dev
```

The frontend will be available at `http://localhost:8080`
The backend API will be available at `http://localhost:3001`

### Build for Production

```bash
npm run build
```

## 🧭 Routes

| Route | Component | Description |
|-------|-----------|-------------|
| `/` | Home | Landing page with features & approach |
| `/video` | VideoDetect | Video deepfake detection interface |
| `/audio` | AudioDetect | Audio deepfake detection interface |
| `/accounts` | Accounts | Login/Register/Profile with auth |
| `/contact` | Contact | Contact form |
| `*` | NotFound | 404 page |

## 🔐 API Endpoints

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| `POST` | `/api/auth/register` | Create new account | No |
| `POST` | `/api/auth/login` | Authenticate user | No |
| `POST` | `/api/auth/logout` | End session | No |
| `GET` | `/api/auth/me` | Get current user | Yes (JWT) |

## 📝 Usage Guide

### Authentication
1. Navigate to `/accounts`
2. Register a new account (Login tab)
3. Or login with existing credentials (Register tab)
4. View profile in Profile tab
5. Logout updates navbar state

### Video Detection
1. Navigate to `/video`
2. Drag & drop or click to upload a video file (.mp4, .mov, .mkv, .webm)
3. Adjust settings (frame sampling rate, face focus)
4. Click "Analyze Video"
5. View results in the result panel (UI simulation only)

### Audio Detection
1. Navigate to `/audio`
2. Upload an audio file (.wav, .mp3, .m4a, .flac)
3. Configure settings (chunk size, spectrogram mode, noise reduction)
4. Click "Analyze Audio"
5. Review waveform, spectrogram, and results

### Theme Toggle
- Click the sun/moon icon in the navbar
- Theme persists in localStorage
- Applies to all components via CSS variables

## 🔒 Important Notes

### Authentication
- SQLite database created automatically on first run
- JWT tokens stored in HTTP-only cookies for security
- Passwords hashed with bcrypt (10 rounds)
- Session persists across page reloads

### Analysis Features
The "Analyze" buttons simulate processing with a 3-second delay. **No actual deepfake detection occurs** - this is a UI demonstration showing how the analysis workflow would function.

### No Fake Metrics
- **No placeholder percentages** or fake confidence scores
- Result panels show labels only ("Decision will appear here after analysis")
- EmptyState components guide users clearly

### File Validation
- **Video**: Max 1GB, types: .mp4, .mov, .mkv, .webm
- **Audio**: Max 100MB, types: .wav, .mp3, .m4a, .flac
- Validation errors shown via toast notifications

## 🎯 Next Steps (Backend Integration)

To make this a fully functional deepfake detector:

1. **Backend API**
   - Python/FastAPI server
   - ResNeXt model loading (PyTorch/TensorFlow)
   - LSTM sequence processing
   - File upload handling

2. **Model Integration**
   - Pre-trained ResNeXt weights
   - LSTM temporal analysis
   - Feature extraction pipeline
   - Confidence scoring

3. **Database** (optional)
   - Store analysis results
   - User authentication
   - Upload history

4. **Deployment**
   - Frontend: Vercel, Netlify, or Cloudflare Pages
   - Backend: AWS Lambda, Google Cloud Run, or dedicated server
   - CDN for static assets

## 🤝 Contributing

This is a demonstration project. For production use:
- Add comprehensive tests (Jest, React Testing Library)
- Implement error boundaries
- Add rate limiting for uploads
- Set up CI/CD pipeline
- Add monitoring & analytics

## 📄 License

This project is for educational and demonstration purposes.

## 🙏 Acknowledgments

Built with:
- shadcn/ui for component inspiration
- Tailwind Labs for design utilities
- Lucide for beautiful icons
- WaveSurfer.js team for audio visualization

---

**Project URL**: https://lovable.dev/projects/ef043070-e7db-4394-a836-8e81cede7c66
