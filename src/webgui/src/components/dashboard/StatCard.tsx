import { Card, CardContent } from "@/components/ui/card";
import { StatItem } from "@/types/dashboard";
import { Type, AlignLeft, Text } from "lucide-react";

const iconMap: Record<string, React.ComponentType<{ className?: string }>> = {
  type: Type,
  "align-left": AlignLeft,
  text: Text,
};

interface StatCardProps {
  stat: StatItem;
}

export function StatCard({ stat }: StatCardProps) {
  const Icon = stat.icon ? iconMap[stat.icon] : null;

  return (
    <Card>
      <CardContent className="pt-6">
        <div className="flex items-center gap-4">
          {Icon && (
            <div className="p-3 bg-primary/10 rounded-lg">
              <Icon className="h-6 w-6 text-primary" />
            </div>
          )}
          <div className="flex-1">
            <p className="text-sm font-medium text-muted-foreground">{stat.label}</p>
            <div className="flex items-baseline gap-1 mt-1">
              <p className="text-2xl font-bold">{stat.value}</p>
              {stat.unit && <span className="text-sm text-muted-foreground">{stat.unit}</span>}
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  );
}