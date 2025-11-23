"use client";

import { useState } from "react";
import { Card } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { MessageSquare, Users, ShieldCheck, BarChart, Sparkles } from "lucide-react";

const features = [
  {
    id: 1,
    title: "Chat Moderation",
    icon: MessageSquare,
    description: "Real-time filtering of toxic messages in chat applications",
    details: "Automatically detect and filter harmful content in live chat conversations. Our AI analyzes messages instantly, flagging toxicity before it reaches your community.",
    stats: { accuracy: "98%", speed: "< 50ms" }
  },
  {
    id: 2,
    title: "Community Forums",
    icon: Users,
    description: "Keep discussion boards healthy and respectful",
    details: "Maintain a positive environment in forums and discussion boards. ToxiShield identifies offensive posts, harassment, and spam automatically.",
    stats: { accuracy: "97%", speed: "< 100ms" }
  },
  {
    id: 3,
    title: "Content Review",
    icon: ShieldCheck,
    description: "Automated content moderation for user-generated content",
    details: "Review and filter user submissions at scale. From comments to reviews, ensure your platform maintains high quality standards.",
    stats: { accuracy: "96%", speed: "< 75ms" }
  },
  {
    id: 4,
    title: "Analytics Dashboard",
    icon: BarChart,
    description: "Track toxicity trends and moderation metrics",
    details: "Gain insights into community health with comprehensive analytics. Monitor toxicity trends, common violations, and moderation effectiveness.",
    stats: { accuracy: "N/A", speed: "Real-time" }
  },
  {
    id: 5,
    title: "Custom Training",
    icon: Sparkles,
    description: "Fine-tune models for your specific use case",
    details: "Customize toxicity detection to match your community guidelines. Train models on your specific content and moderation policies.",
    stats: { accuracy: "99%+", speed: "Custom" }
  }
];

const FeaturesSection = () => {
  const [activeFeature, setActiveFeature] = useState(0);

  return (
    <section id="features" className="py-20 bg-background">
      <div className="container mx-auto px-4 md:px-6">
        <div className="text-center mb-12 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 text-foreground">Powerful Features</h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Comprehensive toxicity detection for every use case
          </p>
        </div>

        <div className="flex flex-wrap justify-center gap-3 mb-8">
          {features.map((feature, index) => (
            <Button
              key={feature.id}
              variant={activeFeature === index ? "default" : "outline"}
              onClick={() => setActiveFeature(index)}
              className={`transition-all duration-300 ${
                activeFeature === index 
                  ? "bg-primary text-primary-foreground shadow-elevated" 
                  : "hover:border-primary/50"
              }`}
            >
              <feature.icon className="w-4 h-4 mr-2" />
              Feature {feature.id}
            </Button>
          ))}
        </div>

        <Card className="p-8 md:p-12 bg-card border-border shadow-elevated animate-scale-in">
          <div className="grid md:grid-cols-2 gap-8 items-center">
            <div>
              <div className="flex items-center gap-3 mb-4">
                {(() => {
                  const IconComponent = features[activeFeature].icon;
                  return (
                    <div className="p-3 bg-primary/10 rounded-lg">
                      <IconComponent className="w-8 h-8 text-primary" />
                    </div>
                  );
                })()}
                <h3 className="text-3xl font-bold text-foreground">{features[activeFeature].title}</h3>
              </div>
              <p className="text-lg text-muted-foreground mb-6">{features[activeFeature].description}</p>
              <p className="text-foreground/80 mb-6">{features[activeFeature].details}</p>
              <div className="flex gap-6">
                <div>
                  <div className="text-2xl font-bold text-primary">{features[activeFeature].stats.accuracy}</div>
                  <div className="text-sm text-muted-foreground">Accuracy</div>
                </div>
                <div>
                  <div className="text-2xl font-bold text-accent">{features[activeFeature].stats.speed}</div>
                  <div className="text-sm text-muted-foreground">Response Time</div>
                </div>
              </div>
            </div>
            <div className="bg-muted/30 rounded-lg p-8 min-h-[300px] flex items-center justify-center border border-border">
              <div className="text-center text-muted-foreground">
                {(() => {
                  const IconComponent = features[activeFeature].icon;
                  return <IconComponent className="w-24 h-24 mx-auto mb-4 opacity-20" />;
                })()}
                <p>Feature visualization</p>
              </div>
            </div>
          </div>
        </Card>
      </div>
    </section>
  );
};

export default FeaturesSection;
