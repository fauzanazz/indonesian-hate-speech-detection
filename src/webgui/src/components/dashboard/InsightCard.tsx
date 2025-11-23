import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert";
import { InsightItem } from "@/types/dashboard";
import { TrendingDown, Check, Twitter, UsersRound, Info, AlertTriangle } from "lucide-react";

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  "trending-down": TrendingDown,
  check: Check,
  twitter: Twitter,
  "users-check": UsersRound,
  info: Info,
  "alert-triangle": AlertTriangle,
};

const severityStyles: Record<string, string> = {
  success: "border-green-500 bg-green-50 text-green-900",
  warning: "border-yellow-500 bg-yellow-50 text-yellow-900",
  info: "border-blue-500 bg-blue-50 text-blue-900",
  danger: "border-red-500 bg-red-50 text-red-900",
};

interface InsightCardProps {
  insight: InsightItem;
}

export function InsightCard({ insight }: InsightCardProps) {
  const Icon = iconMap[insight.icon] || Info;
  const className = severityStyles[insight.severity] || severityStyles.info;

  return (
    <Alert className={className}>
      <Icon className="h-4 w-4" />
      <AlertTitle>{insight.title}</AlertTitle>
      <AlertDescription>{insight.description}</AlertDescription>
    </Alert>
  );
}