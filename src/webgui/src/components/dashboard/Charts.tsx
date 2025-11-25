"use client";

import { ChartConfig } from "@/types/dashboard";
import { PieChart, Pie, Cell, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieLabelRenderProps } from "recharts";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface ChartComponentProps {
  title: string;
  chart: ChartConfig;
}

export function ChartComponent({ title, chart }: ChartComponentProps) {
  const { type, config } = chart;

  const renderChart = () => {
    switch (type) {
      case "donut":
        return <DonutChart config={config} />;
      case "bar-horizontal":
        return <HorizontalBarChart config={config} />;
      case "grouped-bar":
        return <GroupedBarChart config={config} />;
      case "stacked-bar":
        return <StackedBarChart config={config} />;
      default:
        return <div>Unsupported chart type</div>;
    }
  };

  return (
    <Card>
      <CardHeader>
        <CardTitle>{title}</CardTitle>
      </CardHeader>
      <CardContent>
        {renderChart()}
      </CardContent>
    </Card>
  );
}

function DonutChart({ config }: { config: ChartConfig["config"] }) {
  const data = (config.data || []).map(item => ({
    name: item.label || '',
    value: item.value || 0,
    percentage: item.percentage || 0
  }));
  const colors = config.colors || [];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <PieChart>
        <Pie
          data={data}
          cx="50%"
          cy="50%"
          innerRadius={60}
          outerRadius={100}
          paddingAngle={5}
          dataKey="value"
          label={(props: PieLabelRenderProps) => `${((props.percent ?? 0) * 100).toFixed(2)}%`}
        >
          {data.map((_, index) => (
            <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
          ))}
        </Pie>
        <Tooltip />
        <Legend />
      </PieChart>
    </ResponsiveContainer>
  );
}

function HorizontalBarChart({ config }: { config: ChartConfig["config"] }) {
  const data = config.data || [];
  const colors = config.colors || [];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} layout="vertical">
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis type="number" />
        <YAxis dataKey="label" type="category" />
        <Tooltip />
        <Legend />
        <Bar dataKey="value" fill={colors[0]}>
          {data.map((_, index) => (
            <Cell key={`cell-${index}`} fill={colors[index % colors.length]} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function GroupedBarChart({ config }: { config: ChartConfig["config"] }) {
  const data = config.data || [];
  const colors = config.colors || [];
  const groups = config.groups || [];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="metric" />
        <YAxis />
        <Tooltip />
        <Legend />
        {groups.map((group, index) => (
          <Bar 
            key={group} 
            dataKey={group.toLowerCase().replace(/\s+/g, '_')} 
            fill={colors[index % colors.length]} 
            name={group}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}

function StackedBarChart({ config }: { config: ChartConfig["config"] }) {
  const data = config.data || [];
  const colors = config.colors || [];
  const stackGroups = config.stack_groups || [];

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="platform" />
        <YAxis />
        <Tooltip />
        <Legend />
        {stackGroups.map((group, index) => (
          <Bar 
            key={group} 
            dataKey={group.toLowerCase().replace(/\s+/g, '_')} 
            stackId="a" 
            fill={colors[index % colors.length]}
            name={group}
          />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}