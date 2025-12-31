import { Link } from 'react-router-dom';
import { Shield, Layers, Brain, Lock, Search, Video, Music, Upload, Scan, CheckCircle2, ArrowRight } from 'lucide-react';
import { Button } from '@/components/Button';
import { Card, CardContent } from '@/components/Card';

export default function Home() {
  const features = [
    {
      icon: Layers,
      title: 'Feature Extraction',
      description: 'ResNeXt neural networks analyze video frames and audio spectrograms to extract deep features',
    },
    {
      icon: Brain,
      title: 'Temporal Analysis',
      description: 'LSTM sequences capture temporal patterns that distinguish authentic content from deepfakes',
    },
    {
      icon: CheckCircle2,
      title: 'Decision & Confidence',
      description: 'Clear results with confidence metrics to help you make informed decisions about media authenticity',
    },
  ];

  const whyMatters = [
    { icon: Shield, text: 'Combat misinformation and protect media integrity' },
    { icon: Lock, text: 'Safeguard personal and organizational reputation' },
    { icon: Search, text: 'Support forensic investigations and legal proceedings' },
  ];

  const pipeline = [
    { icon: Upload, label: 'Upload' },
    { icon: Scan, label: 'Preprocess' },
    { icon: Layers, label: 'ResNeXt' },
    { icon: Brain, label: 'LSTM' },
    { icon: CheckCircle2, label: 'Prediction' },
  ];

  return (
    <div className="w-full">
      {/* Hero Section */}
      <section className="relative py-20 md:py-32 overflow-hidden">
        <div className="absolute inset-0 bg-gradient-hero animate-gradient-shift opacity-10" />
        <div className="container mx-auto px-4 relative">
          <div className="max-w-4xl mx-auto text-center space-y-8 animate-fade-in">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass-card">
              <Shield className="w-4 h-4 text-primary" />
              <span className="text-sm font-medium">AI-Powered Detection</span>
            </div>
            
            <h1 className="text-5xl md:text-7xl font-bold leading-tight">
              <span className="bg-gradient-to-r from-primary via-secondary to-primary bg-clip-text text-transparent">
                DeepShield
              </span>
            </h1>
            
            <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
              ResNeXt + LSTM powered detection for video & audio deepfakes
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center pt-4">
              <Link to="/video">
                <Button variant="hero" size="lg" className="w-full sm:w-auto">
                  <Video className="w-5 h-5 mr-2" />
                  Try Video Detection
                </Button>
              </Link>
              <Link to="/audio">
                <Button variant="secondary" size="lg" className="w-full sm:w-auto">
                  <Music className="w-5 h-5 mr-2" />
                  Try Audio Detection
                </Button>
              </Link>
            </div>
            
            <a
              href="#approach"
              className="inline-flex items-center gap-2 text-sm text-muted-foreground hover:text-primary transition-colors"
            >
              Learn more
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-20 bg-muted/30">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">How It Works</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              Our dual-stage architecture combines spatial and temporal analysis for robust deepfake detection
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8 max-w-6xl mx-auto">
            {features.map((feature, idx) => (
              <Card key={idx} glass className="animate-scale-in" style={{ animationDelay: `${idx * 100}ms` }}>
                <CardContent className="pt-6">
                  <div className="w-12 h-12 rounded-xl bg-gradient-hero flex items-center justify-center mb-4">
                    <feature.icon className="w-6 h-6 text-white" />
                  </div>
                  <h3 className="text-xl font-bold mb-2">{feature.title}</h3>
                  <p className="text-muted-foreground">{feature.description}</p>
                </CardContent>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* Why It Matters */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="grid lg:grid-cols-2 gap-12 items-center max-w-6xl mx-auto">
            <div>
              <h2 className="text-3xl md:text-4xl font-bold mb-6">Why It Matters</h2>
              <p className="text-lg text-muted-foreground mb-8">
                Deepfakes pose serious threats to trust, security, and truth in our digital age. 
                Our detection system helps protect against manipulation and misinformation.
              </p>
              <div className="space-y-4">
                {whyMatters.map((item, idx) => (
                  <div key={idx} className="flex items-start gap-4">
                    <div className="w-10 h-10 rounded-lg bg-primary/10 flex items-center justify-center flex-shrink-0">
                      <item.icon className="w-5 h-5 text-primary" />
                    </div>
                    <p className="text-foreground pt-2">{item.text}</p>
                  </div>
                ))}
              </div>
            </div>
            
            <Card gradient className="p-8">
              <h3 className="text-2xl font-bold mb-4">Detection Capabilities</h3>
              <ul className="space-y-3">
                <li className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0" />
                  <span>Face-swap and face-reenactment detection</span>
                </li>
                <li className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0" />
                  <span>Voice cloning and audio synthesis detection</span>
                </li>
                <li className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0" />
                  <span>Temporal inconsistency analysis</span>
                </li>
                <li className="flex items-center gap-3">
                  <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0" />
                  <span>Multi-modal feature extraction</span>
                </li>
              </ul>
            </Card>
          </div>
        </div>
      </section>

      {/* Approach Timeline */}
      <section id="approach" className="py-20 bg-muted/30">
        <div className="container mx-auto px-4">
          <div className="text-center mb-12">
            <h2 className="text-3xl md:text-4xl font-bold mb-4">Our Approach</h2>
            <p className="text-muted-foreground max-w-2xl mx-auto">
              A streamlined pipeline from upload to prediction
            </p>
          </div>
          
          <div className="flex flex-wrap justify-center items-center gap-4 max-w-4xl mx-auto">
            {pipeline.map((step, idx) => (
              <div key={idx} className="flex items-center">
                <Card glass className="w-32 h-32 flex flex-col items-center justify-center">
                  <step.icon className="w-8 h-8 text-primary mb-2" />
                  <span className="text-sm font-medium">{step.label}</span>
                </Card>
                {idx < pipeline.length - 1 && (
                  <ArrowRight className="w-6 h-6 text-muted-foreground mx-2" />
                )}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="grid md:grid-cols-2 gap-6 max-w-4xl mx-auto">
            <Link to="/video">
              <Card glass className="p-8 hover:scale-105 transition-transform cursor-pointer h-full">
                <Video className="w-12 h-12 text-primary mb-4" />
                <h3 className="text-2xl font-bold mb-2">Video Detection</h3>
                <p className="text-muted-foreground mb-4">
                  Analyze video files for face manipulation and synthetic content
                </p>
                <div className="flex items-center text-primary font-medium">
                  Get Started
                  <ArrowRight className="w-4 h-4 ml-2" />
                </div>
              </Card>
            </Link>
            
            <Link to="/audio">
              <Card glass className="p-8 hover:scale-105 transition-transform cursor-pointer h-full">
                <Music className="w-12 h-12 text-secondary mb-4" />
                <h3 className="text-2xl font-bold mb-2">Audio Detection</h3>
                <p className="text-muted-foreground mb-4">
                  Detect voice cloning and synthetic audio generation
                </p>
                <div className="flex items-center text-secondary font-medium">
                  Get Started
                  <ArrowRight className="w-4 h-4 ml-2" />
                </div>
              </Card>
            </Link>
          </div>
        </div>
      </section>

      {/* Design Principles */}
      <section className="py-20 bg-muted/30">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto text-center">
            <h2 className="text-3xl md:text-4xl font-bold mb-8">Design Principles</h2>
            <div className="grid md:grid-cols-3 gap-8">
              <div>
                <Lock className="w-8 h-8 text-primary mx-auto mb-3" />
                <h3 className="font-bold mb-2">Privacy-First</h3>
                <p className="text-sm text-muted-foreground">
                  Your media files are processed securely with no permanent storage
                </p>
              </div>
              <div>
                <Search className="w-8 h-8 text-primary mx-auto mb-3" />
                <h3 className="font-bold mb-2">Explainable</h3>
                <p className="text-sm text-muted-foreground">
                  Clear visualization of analysis results and confidence metrics
                </p>
              </div>
              <div>
                <Shield className="w-8 h-8 text-primary mx-auto mb-3" />
                <h3 className="font-bold mb-2">Robust</h3>
                <p className="text-sm text-muted-foreground">
                  Tested against diverse deepfake techniques and datasets
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
