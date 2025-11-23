import { loadDashboardConfig } from "@/lib/dashboard-parser";
import { MetricCard } from "@/components/dashboard/MetricCard";
import { StatCard } from "@/components/dashboard/StatCard";
import { InsightCard } from "@/components/dashboard/InsightCard";
import { ChartComponent } from "@/components/dashboard/Charts";
import { DatasetSearch } from "@/components/dataset/DatasetSearch";

export default function DatasetEDAPage() {
  const config = loadDashboardConfig();

  const getLayoutClass = (layout: string) => {
    switch (layout) {
      case "grid-4":
        return "grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4";
      case "half-width":
        return "w-full lg:w-1/2";
      case "full-width":
        return "w-full";
      default:
        return "w-full";
    }
  };

  return (
    <div className="min-h-screen bg-background">
      <div className="container mx-auto py-8 px-4" style={{ maxWidth: config.config.layout.max_width }}>
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold mb-2">{config.dashboard.title}</h1>
          <p className="text-muted-foreground">{config.dashboard.description}</p>
        </div>

        {/* Dataset Search Section */}
        <div className="mb-8">
          <DatasetSearch />
        </div>

        {/* Sections */}
        {config.sections.map((section) => (
          <div key={section.id} className="mb-8">
            <h2 className="text-2xl font-semibold mb-4">{section.title}</h2>
            
            {/* Metrics Section */}
            {section.type === "metrics" && section.metrics && (
              <div className={getLayoutClass(section.layout)}>
                {section.metrics.map((metric, index) => (
                  <MetricCard key={index} metric={metric} />
                ))}
              </div>
            )}

            {/* Stats Grid Section */}
            {section.type === "stats-grid" && section.stats && (
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                {section.stats.map((stat, index) => (
                  <StatCard key={index} stat={stat} />
                ))}
              </div>
            )}

            {/* Visualization Section */}
            {section.type === "visualization" && section.chart && (
              <div className={section.layout === "half-width" ? "inline-block w-full lg:w-1/2 pr-2" : "w-full"}>
                <ChartComponent title={section.title} chart={section.chart} />
              </div>
            )}

            {/* Insights Section */}
            {section.type === "insights" && section.insights && (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {section.insights.map((insight, index) => (
                  <InsightCard key={index} insight={insight} />
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}