import * as TOML from '@iarna/toml';
import { DashboardConfig } from '@/types/dashboard';

export async function parseDashboardConfig(tomlPath: string): Promise<DashboardConfig> {
  const response = await fetch(tomlPath);
  const tomlText = await response.text();
  const parsed = TOML.parse(tomlText) as unknown as DashboardConfig;
  return parsed;
}

export function loadDashboardConfig(): DashboardConfig {
  // Hardcoded config based on the TOML plan
  return {
    dashboard: {
      title: "Indonesian Hate Speech Dataset Analysis",
      description: "Comprehensive analysis of hate speech detection dataset from Twitter and Instagram",
      version: "1.0.0"
    },
    sections: [
      {
        id: "overview",
        title: "Dataset Overview",
        type: "metrics",
        layout: "grid-4",
        metrics: [
          { label: "Total Samples", value: 14306, icon: "database" },
          { label: "Hate Speech", value: "42.29%", trend: "neutral", icon: "alert-triangle" },
          { label: "Non-Hate Speech", value: "57.71%", trend: "neutral", icon: "check-circle" },
          { label: "Annotators", value: 3, description: "Quality control standard", icon: "users" }
        ]
      },
      {
        id: "class-distribution",
        title: "Label Distribution",
        type: "visualization",
        layout: "half-width",
        chart: {
          type: "donut",
          data_source: "label_counts",
          config: {
            colors: ["#ef4444", "#10b981"],
            labels: ["Hate Speech", "Non-Hate Speech"],
            show_percentage: true,
            center_text: "Balanced",
            data: [
              { label: "Hate Speech", value: 6050, percentage: 42.29 },
              { label: "Non-Hate Speech", value: 8256, percentage: 57.71 }
            ]
          }
        }
      },
      {
        id: "source-breakdown",
        title: "Data Source Distribution",
        type: "visualization",
        layout: "half-width",
        chart: {
          type: "bar-horizontal",
          data_source: "source_counts",
          config: {
            colors: ["#3b82f6", "#a855f7"],
            show_values: true,
            show_percentage: true,
            data: [
              { label: "Twitter", value: 13734, percentage: 96.0 },
              { label: "Instagram", value: 572, percentage: 4.0 }
            ]
          }
        }
      },
      {
        id: "text-metrics",
        title: "Text Characteristics",
        type: "stats-grid",
        layout: "full-width",
        stats: [
          { label: "Avg Character Length", value: "112.59", unit: "chars", icon: "type" },
          { label: "Avg Word Count", value: "17.02", unit: "words", icon: "align-left" },
          { label: "Avg Word Length", value: "6.66", unit: "chars/word", icon: "text" }
        ]
      },
      {
        id: "comparative-analysis",
        title: "Hate Speech vs Non-Hate Speech Characteristics",
        type: "visualization",
        layout: "full-width",
        chart: {
          type: "grouped-bar",
          data_source: "label_comparison",
          config: {
            x_axis: "Metric",
            y_axis: "Value",
            groups: ["Hate Speech", "Non-Hate Speech"],
            colors: ["#ef4444", "#10b981"],
            data: [
              { metric: "Avg Word Count", hate_speech: 15.63, non_hate_speech: 18.03 }
            ]
          }
        }
      },
      {
        id: "key-insights",
        title: "Key Insights",
        type: "insights",
        layout: "full-width",
        insights: [
          {
            icon: "trending-down",
            title: "Shorter Hate Speech",
            description: "Hate speech texts are 2.40 words shorter on average (15.63 vs 18.03 words)",
            severity: "warning"
          },
          {
            icon: "check",
            title: "Balanced Dataset",
            description: "Near-balanced distribution (42.29% vs 57.71%) - no aggressive resampling needed",
            severity: "success"
          },
          {
            icon: "twitter",
            title: "Twitter Dominant",
            description: "96% of data from Twitter, 4% from Instagram - platform bias consideration needed",
            severity: "info"
          },
          {
            icon: "users-check",
            title: "Quality Annotated",
            description: "All samples annotated by 3 annotators for reliability and consistency",
            severity: "success"
          }
        ]
      },
      {
        id: "platform-distribution",
        title: "Label Distribution by Platform",
        type: "visualization",
        layout: "full-width",
        chart: {
          type: "stacked-bar",
          data_source: "platform_label_distribution",
          config: {
            x_axis: "Platform",
            y_axis: "Percentage",
            stack_groups: ["Hate Speech", "Non-Hate Speech"],
            colors: ["#ef4444", "#10b981"],
            show_percentage: true,
            data: [
              { platform: "Twitter", hate_speech: 42.29, non_hate_speech: 57.71 },
              { platform: "Instagram", hate_speech: 42.0, non_hate_speech: 58.0, note: "Proportional distribution" }
            ]
          }
        }
      }
    ],
    config: {
      theme: "light",
      refresh_interval: 0,
      responsive: true,
      export_formats: ["pdf", "png", "json"],
      colors: {
        primary: "#3b82f6",
        success: "#10b981",
        warning: "#f59e0b",
        danger: "#ef4444",
        neutral: "#6b7280"
      },
      layout: {
        max_width: "1400px",
        padding: "24px",
        gap: "16px"
      }
    }
  };
}