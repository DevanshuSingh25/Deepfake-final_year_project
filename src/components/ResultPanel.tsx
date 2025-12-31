import { AlertCircle } from 'lucide-react';
import { Card, CardHeader, CardTitle, CardContent } from './Card';
import { EmptyState } from './EmptyState';

interface ResultPanelProps {
  hasResult: boolean;
}

export function ResultPanel({ hasResult }: ResultPanelProps) {
  if (!hasResult) {
    return (
      <Card>
        <CardContent className="py-8">
          <EmptyState
            icon={AlertCircle}
            title="No analysis yet"
            description="Upload a file and click 'Analyze' to see the detection results"
          />
        </CardContent>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>Analysis Results</CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        <div>
          <h4 className="text-sm font-medium text-muted-foreground mb-2">
            Overall Decision
          </h4>
          <div className="p-4 rounded-lg bg-muted">
            <p className="text-sm text-muted-foreground">
              Decision will appear here after analysis
            </p>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-medium text-muted-foreground mb-2">
            Confidence
          </h4>
          <div className="p-4 rounded-lg bg-muted">
            <p className="text-sm text-muted-foreground">
              Confidence metrics will appear here
            </p>
          </div>
        </div>

        <div>
          <h4 className="text-sm font-medium text-muted-foreground mb-2">
            Notes
          </h4>
          <div className="p-4 rounded-lg bg-muted">
            <p className="text-sm text-muted-foreground">
              Additional analysis notes will appear here
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}
