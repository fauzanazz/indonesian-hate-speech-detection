import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { MetricCard as MetricCardType } from "@/types/dashboard";
import { Database, AlertTriangle, CheckCircle, Users, TrendingUp, TrendingDown } from "lucide-react";

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  database: Database,
  "alert-triangle": AlertTriangle,
  "check-circle": CheckCircle,
  users: Users,
  "trending-up": TrendingUp,
  "trending-down": TrendingDown,
};

interface MetricCardProps {
  metric: MetricCardType;
}

export function MetricCard({ metric }: MetricCardProps) {
  const Icon = metric.icon ? iconMap[metric.icon] : null;

  return (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{metric.label}</CardTitle>
        {Icon && <Icon className="h-4 w-4 text-muted-foreground" />}
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{metric.value}</div>
        {metric.description && (
          <p className="text-xs text-muted-foreground mt-1">{metric.description}</p>
        )}
      </CardContent>
    </Card>
  );
}