import { useState, FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { Shield, Lock, Smartphone, User, Mail } from 'lucide-react';
import { Card } from '@/components/Card';
import { Tabs } from '@/components/Tabs';
import { Button } from '@/components/Button';
import { Toast, ToastType } from '@/components/Toast';
import { useAuth } from '@/context/AuthContext';

export default function Accounts() {
  const [toast, setToast] = useState<{ type: ToastType; message: string } | null>(null);
  const { user, login, register, logout } = useAuth();
  const navigate = useNavigate();

  const handleLogin = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const email = formData.get('email') as string;
    const password = formData.get('password') as string;

    try {
      await login(email, password);
      setToast({ type: 'success', message: 'Login successful!' });
      setTimeout(() => navigate('/'), 1500);
    } catch (error) {
      setToast({ type: 'error', message: error instanceof Error ? error.message : 'Login failed' });
    }
  };

  const handleRegister = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);
    const name = formData.get('name') as string;
    const email = formData.get('email') as string;
    const password = formData.get('password') as string;
    const confirmPassword = formData.get('confirmPassword') as string;

    if (password !== confirmPassword) {
      setToast({ type: 'error', message: 'Passwords do not match' });
      return;
    }

    try {
      await register(name, email, password);
      setToast({ type: 'success', message: 'Registration successful!' });
      setTimeout(() => navigate('/'), 1500);
    } catch (error) {
      setToast({ type: 'error', message: error instanceof Error ? error.message : 'Registration failed' });
    }
  };

  const handleLogout = async () => {
    try {
      await logout();
      setToast({ type: 'success', message: 'Logged out successfully' });
      setTimeout(() => navigate('/'), 1500);
    } catch (error) {
      setToast({ type: 'error', message: 'Logout failed' });
    }
  };

  const loginForm = (
    <form onSubmit={handleLogin} className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">Email</label>
        <input
          type="email"
          name="email"
          required
          className="w-full px-4 py-3 rounded-lg bg-background border border-input focus:ring-2 focus:ring-ring focus:border-ring"
          placeholder="your@email.com"
        />
      </div>
      <div>
        <label className="block text-sm font-medium mb-2">Password</label>
        <input
          type="password"
          name="password"
          required
          className="w-full px-4 py-3 rounded-lg bg-background border border-input focus:ring-2 focus:ring-ring focus:border-ring"
          placeholder="••••••••"
        />
      </div>
      <div className="flex items-center">
        <input
          type="checkbox"
          id="remember"
          className="w-4 h-4 rounded border-input focus:ring-2 focus:ring-ring"
        />
        <label htmlFor="remember" className="ml-2 text-sm text-muted-foreground">
          Remember me
        </label>
      </div>
      <Button type="submit" className="w-full" size="lg">
        Sign In
      </Button>
    </form>
  );

  const registerForm = (
    <form onSubmit={handleRegister} className="space-y-4">
      <div>
        <label className="block text-sm font-medium mb-2">Name</label>
        <input
          type="text"
          name="name"
          required
          minLength={2}
          maxLength={100}
          className="w-full px-4 py-3 rounded-lg bg-background border border-input focus:ring-2 focus:ring-ring focus:border-ring"
          placeholder="Your Name"
        />
      </div>
      <div>
        <label className="block text-sm font-medium mb-2">Email</label>
        <input
          type="email"
          name="email"
          required
          maxLength={255}
          className="w-full px-4 py-3 rounded-lg bg-background border border-input focus:ring-2 focus:ring-ring focus:border-ring"
          placeholder="your@email.com"
        />
      </div>
      <div>
        <label className="block text-sm font-medium mb-2">Password</label>
        <input
          type="password"
          name="password"
          required
          minLength={8}
          maxLength={100}
          className="w-full px-4 py-3 rounded-lg bg-background border border-input focus:ring-2 focus:ring-ring focus:border-ring"
          placeholder="••••••••"
        />
      </div>
      <div>
        <label className="block text-sm font-medium mb-2">Confirm Password</label>
        <input
          type="password"
          name="confirmPassword"
          required
          minLength={8}
          maxLength={100}
          className="w-full px-4 py-3 rounded-lg bg-background border border-input focus:ring-2 focus:ring-ring focus:border-ring"
          placeholder="••••••••"
        />
      </div>
      <Button type="submit" className="w-full" size="lg">
        Create Account
      </Button>
    </form>
  );

  const profileForm = (
    <div className="space-y-4">
      <div className="flex justify-center mb-6">
        <div className="w-24 h-24 rounded-full bg-gradient-hero flex items-center justify-center">
          <User className="w-12 h-12 text-white" />
        </div>
      </div>
      <div>
        <label className="block text-sm font-medium mb-2">Name</label>
        <input
          type="text"
          readOnly
          value={user?.name || ''}
          className="w-full px-4 py-3 rounded-lg bg-muted border border-input cursor-not-allowed"
        />
      </div>
      <div>
        <label className="block text-sm font-medium mb-2">Email</label>
        <input
          type="email"
          readOnly
          value={user?.email || ''}
          className="w-full px-4 py-3 rounded-lg bg-muted border border-input cursor-not-allowed"
        />
      </div>
      <Button onClick={handleLogout} className="w-full" size="lg" variant="outline">
        Logout
      </Button>
    </div>
  );

  const tabs = [
    { label: 'Login', content: loginForm },
    { label: 'Register', content: registerForm },
    { label: 'Profile', content: profileForm },
  ];

  return (
    <div className="min-h-screen py-12 bg-gradient-to-b from-background to-muted/30">
      <div className="container mx-auto px-4">
        <div className="max-w-5xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-8 items-start">
            {/* Left Panel - Brand Promise */}
            <div className="lg:sticky lg:top-24">
              <Card className="bg-gradient-hero text-white p-8 lg:p-12">
                <Shield className="w-16 h-16 mb-6" />
                <h2 className="text-3xl font-bold mb-4">Secure Access</h2>
                <p className="text-lg mb-8 opacity-90">
                  Join DeepShield to access advanced detection features and manage your analysis history
                </p>
                <div className="space-y-4">
                  <div className="flex items-center gap-3">
                    <Lock className="w-5 h-5" />
                    <span>Secure uploads & encrypted results</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Shield className="w-5 h-5" />
                    <span>Privacy-first design</span>
                  </div>
                  <div className="flex items-center gap-3">
                    <Smartphone className="w-5 h-5" />
                    <span>One account, multi-device access</span>
                  </div>
                </div>
              </Card>
            </div>

            {/* Right Panel - Auth Forms */}
            <Card glass>
              <div className="p-6">
                <div className="flex items-center gap-2 mb-6">
                  <Mail className="w-5 h-5 text-primary" />
                  <h2 className="text-2xl font-bold">Account</h2>
                </div>
                <Tabs tabs={tabs} />
              </div>
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
