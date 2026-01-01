import { lazy, Suspense } from 'react';
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import { AppLayout } from './layouts/AppLayout';

// Lazy load pages for code splitting
const Home = lazy(() => import('./pages/Home'));
const VideoDetect = lazy(() => import('./pages/VideoDetect'));
const AudioDetect = lazy(() => import('./pages/AudioDetect'));
const Accounts = lazy(() => import('./pages/Accounts'));
const Contact = lazy(() => import('./pages/Contact'));
const NotFound = lazy(() => import('./pages/NotFound'));

// Loading component
function PageLoader() {
  return (
    <div className="flex items-center justify-center min-h-screen">
      <div className="w-12 h-12 border-4 border-primary border-t-transparent rounded-full animate-spin" />
    </div>
  );
}

const router = createBrowserRouter([
  {
    path: '/',
    element: <AppLayout />,
    children: [
      {
        index: true,
        element: (
          <Suspense fallback={<PageLoader />}>
            <Home />
          </Suspense>
        ),
      },
      {
        path: 'video',
        element: (
          <Suspense fallback={<PageLoader />}>
            <VideoDetect />
          </Suspense>
        ),
      },
      {
        path: 'audio',
        element: (
          <Suspense fallback={<PageLoader />}>
            <AudioDetect />
          </Suspense>
        ),
      },
      {
        path: 'accounts',
        element: (
          <Suspense fallback={<PageLoader />}>
            <Accounts />
          </Suspense>
        ),
      },
      {
        path: 'contact',
        element: (
          <Suspense fallback={<PageLoader />}>
            <Contact />
          </Suspense>
        ),
      },
      {
        path: '*',
        element: (
          <Suspense fallback={<PageLoader />}>
            <NotFound />
          </Suspense>
        ),
      },
    ],
  },
]);

export function Router() {
  return <RouterProvider router={router} />;
}
