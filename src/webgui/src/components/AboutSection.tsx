import { Card } from "@/components/ui/card";
import { Avatar, AvatarFallback, AvatarImage } from "@/components/ui/avatar";

const team = [
  { name: "Shabrina Maharani", role: "13522134", initials: "SM", feature: "Creator of Counter-Speech Generation", imageUrl: "https://i.pravatar.cc/150?u=shabrina" },
  { name: "Muhammad Fauzan Azhim", role: "13522153", initials: "MFA", feature: "Creator of Search and Filter", imageUrl: "https://i.pravatar.cc/150?u=fauzan" },
  { name: "Yasmin Farisah Salma", role: "13522140", initials: "YFS", feature: "Creator of Toxicity Detection", imageUrl: "https://i.pravatar.cc/150?u=yasmin" }
];

const AboutSection = () => {
  return (
    <section id="about" className="py-20 bg-muted/30">
      <div className="container mx-auto px-4 md:px-6">
        <div className="text-center mb-12 animate-fade-in">
          <h2 className="text-4xl md:text-5xl font-bold mb-4 text-foreground">About Us</h2>
          <p className="text-xl text-muted-foreground max-w-2xl mx-auto">
            Meet the team behind ToxiShield
          </p>
        </div>

        <div className="max-w-4xl mx-auto">
          <div className="flex justify-center gap-8 mb-12">
            {team.map((member, index) => (
              <div key={index} className="flex flex-col items-center gap-3 animate-fade-in">
                <Avatar className="w-32 h-32 border-4 border-accent/20">
                  <AvatarImage src={member.imageUrl} alt={member.name} />
                  <AvatarFallback className="bg-gradient-accent text-2xl font-bold text-accent-foreground">
                    {member.initials}
                  </AvatarFallback>
                </Avatar>
                <div className="text-center">
                  <h3 className="font-semibold text-foreground">{member.name}</h3>
                  <p className="text-sm text-muted-foreground">{member.role}</p>
                  <p className="text-sm text-muted-foreground">{member.feature}</p>
                </div>
              </div>
            ))}
          </div>

          <Card className="p-8 bg-card border-border shadow-elevated">
            <p className="text-lg text-foreground/80 leading-relaxed">
              ToxiShield was founded in 2023 by a team of AI researchers and community builders who recognized the growing need for effective content moderation. Our mission is to make online spaces safer and more inclusive through cutting-edge machine learning technology.
            </p>
            <p className="text-lg text-foreground/80 leading-relaxed mt-4">
              We believe that healthy online communities are built on respect and safety. Our toxicity detection models are trained on diverse datasets and continuously improved to catch emerging patterns of harmful content while respecting context and nuance.
            </p>
            <p className="text-lg text-foreground/80 leading-relaxed mt-4">
              Today, ToxiShield protects millions of users across social platforms, gaming communities, and enterprise applications worldwide.
            </p>
          </Card>
        </div>
      </div>
    </section>
  );
};

export default AboutSection;
