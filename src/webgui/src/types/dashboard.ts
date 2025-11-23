export interface DashboardConfig {
  dashboard: {
    title: string;
    description: string;
    version: string;
  };
  sections: DashboardSection[];
  config: {
    theme: string;
    refresh_interval: number;
    responsive: boolean;
    export_formats: string[];
    colors: {
      primary: string;
      success: string;
      warning: string;
      danger: string;
      neutral: string;
    };
    layout: {
      max_width: string;
      padding: string;
      gap: string;
    };
  };
}

export interface DashboardSection {
  id: string;
  title: string;
  type: 'metrics' | 'visualization' | 'stats-grid' | 'insights';
  layout: string;
  metrics?: MetricCard[];
  chart?: ChartConfig;
  stats?: StatItem[];
  insights?: InsightItem[];
}

export interface MetricCard {
  label: string;
  value: string | number;
  icon?: string;
  trend?: 'up' | 'down' | 'neutral';
  description?: string;
}

export interface ChartConfig {
  type: 'donut' | 'bar-horizontal' | 'grouped-bar' | 'stacked-bar';
  data_source: string;
  config: {
    colors?: string[];
    labels?: string[];
    show_percentage?: boolean;
    show_values?: boolean;
    center_text?: string;
    x_axis?: string;
    y_axis?: string;
    groups?: string[];
    stack_groups?: string[];
    data?: ChartDataPoint[];
  };
}

export interface ChartDataPoint {
  label?: string;
  value?: number;
  percentage?: number;
  metric?: string;
  hate_speech?: number;
  non_hate_speech?: number;
  platform?: string;
  highlight?: boolean;
  note?: string;
}

export interface StatItem {
  label: string;
  value: string;
  unit?: string;
  icon?: string;
}

export interface InsightItem {
  icon: string;
  title: string;
  description: string;
  severity: 'success' | 'warning' | 'info' | 'danger';
}