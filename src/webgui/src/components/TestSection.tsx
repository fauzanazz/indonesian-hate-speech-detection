"use client";

import { useState } from "react";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import ToxicityTest from "@/components/ToxicityTest";
import SearchTest from "@/components/SearchTest";

const TestSection = () => {
  const [activeTab, setActiveTab] = useState<"toxicity" | "search">("toxicity");

  return (
    <section id="test" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4 md:px-6">
        <div className="text-center mb-12 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 text-foreground">Test</h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Try our toxicity detection and semantic search in action
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <Tabs value={activeTab} onValueChange={(value) => setActiveTab(value as typeof activeTab)}>
            <TabsList className="grid w-full grid-cols-2 mb-6">
              <TabsTrigger value="toxicity">Toxicity Detection</TabsTrigger>
              <TabsTrigger value="search">Semantic Search</TabsTrigger>
            </TabsList>
            
            <TabsContent value="toxicity">
              <ToxicityTest />
            </TabsContent>
            
            <TabsContent value="search">
              <SearchTest />
            </TabsContent>
          </Tabs>
        </div>
      </div>
    </section>
  );
};

export default TestSection;
