import { useState } from 'react';
import { Mail, MessageSquare, FileText } from 'lucide-react';
import { Card, CardContent } from '@/components/Card';
import { Button } from '@/components/Button';
import { Toast, ToastType } from '@/components/Toast';

export default function Contact() {
  const [toast, setToast] = useState<{ type: ToastType; message: string } | null>(null);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setToast({ type: 'warning', message: 'Contact form - Backend integration coming soon' });
  };

  const contacts = [
    {
      icon: Mail,
      title: 'Email',
      description: 'Send us an email',
      link: 'mailto:contact@deepfakeguard.ai',
      linkText: 'contact@deepfakeguard.ai',
    },
    {
      icon: FileText,
      title: 'Documentation',
      description: 'Read the docs',
      link: '#docs',
      linkText: 'Technical Documentation',
    },
  ];

  return (
    <div className="min-h-screen py-12 bg-gradient-to-b from-background to-muted/30">
      <div className="container mx-auto px-4">
        <div className="max-w-5xl mx-auto">
          {/* Header */}
          <div className="text-center mb-12">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card mb-6">
              <MessageSquare className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium">Get in Touch</span>
            </div>
            <h1 className="text-4xl md:text-5xl font-bold mb-4">Contact Us</h1>
            <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
              Have questions, feedback, or collaboration ideas? We'd love to hear from you
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8">
            {/* Contact Form */}
            <Card glass>
              <CardContent className="pt-6">
                <h2 className="text-2xl font-bold mb-6">Send us a message</h2>
                <form onSubmit={handleSubmit} className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Name</label>
                    <input
                      type="text"
                      required
                      minLength={2}
                      maxLength={100}
                      className="w-full px-4 py-3 rounded-lg bg-background border border-input focus:ring-2 focus:ring-ring focus:border-ring"
                      placeholder="Your name"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Email</label>
                    <input
                      type="email"
                      required
                      maxLength={255}
                      className="w-full px-4 py-3 rounded-lg bg-background border border-input focus:ring-2 focus:ring-ring focus:border-ring"
                      placeholder="your@email.com"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium mb-2">Message</label>
                    <textarea
                      required
                      minLength={10}
                      maxLength={1000}
                      rows={6}
                      className="w-full px-4 py-3 rounded-lg bg-background border border-input focus:ring-2 focus:ring-ring focus:border-ring resize-none"
                      placeholder="Your message..."
                    />
                  </div>
                  <Button type="submit" className="w-full" size="lg">
                    Send Message
                  </Button>
                </form>
              </CardContent>
            </Card>

            {/* Contact Cards */}
            <div className="space-y-6">
              {contacts.map((contact, idx) => (
                <Card key={idx} className="hover:scale-105 transition-transform">
                  <CardContent className="pt-6">
                    <div className="flex items-start gap-4">
                      <div className="w-12 h-12 rounded-xl bg-gradient-hero flex items-center justify-center flex-shrink-0">
                        <contact.icon className="w-6 h-6 text-white" />
                      </div>
                      <div className="flex-1">
                        <h3 className="text-xl font-bold mb-1">{contact.title}</h3>
                        <p className="text-sm text-muted-foreground mb-2">
                          {contact.description}
                        </p>
                        <a
                          href={contact.link}
                          className="text-primary hover:underline text-sm font-medium"
                        >
                          {contact.linkText}
                        </a>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
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
